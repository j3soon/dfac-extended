import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.ddn import DDNMixer
from modules.mixers.dmix import DMixer
from modules.mixers.dplex import DPLEXMixer
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.optim import Adam
import numpy as np


class IQNLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "ddn":
                self.mixer = DDNMixer(args)
            elif args.mixer == "dmix":
                self.mixer = DMixer(args)
            elif args.mixer == "dplex":
                self.mixer = DPLEXMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if args.optimizer == "RMSProp":
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif args.optimizer == "Adam":
            self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        else:
            raise ValueError("Unknown Optimizer")

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        episode_length = rewards.shape[1]
        assert rewards.shape == (batch.batch_size, episode_length, 1)
        actions = batch["actions"][:, :-1]
        assert actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        assert mask.shape == (batch.batch_size, episode_length, 1)
        avail_actions = batch["avail_actions"]
        assert avail_actions.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions)
        actions_onehot = batch["actions_onehot"][:, :-1]

        # Mix
        if self.mixer is not None:
            # Same quantile for quantile mixture
            n_quantile_groups = 1
        else:
            n_quantile_groups = self.args.n_agents

        # Calculate estimated Q-Values
        mac_out = []
        rnd_quantiles = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, agent_rnd_quantiles = self.mac.forward(batch, t=t, forward_type="policy")
            assert agent_outs.shape == (batch.batch_size * self.args.n_agents, self.args.n_actions, self.n_quantiles)
            assert agent_rnd_quantiles.shape == (batch.batch_size * n_quantile_groups, self.n_quantiles)
            agent_rnd_quantiles = agent_rnd_quantiles.view(batch.batch_size, n_quantile_groups, self.n_quantiles)
            rnd_quantiles.append(agent_rnd_quantiles)
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            mac_out.append(agent_outs)
        del agent_outs
        del agent_rnd_quantiles
        mac_out = th.stack(mac_out, dim=1) # Concat over time
        rnd_quantiles = th.stack(rnd_quantiles, dim=1) # Concat over time
        assert mac_out.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
        assert rnd_quantiles.shape == (batch.batch_size, episode_length+1, n_quantile_groups, self.n_quantiles)
        rnd_quantiles = rnd_quantiles[:,:-1]
        assert rnd_quantiles.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)

        # Pick the Q-Values for the actions taken by each agent
        actions_for_quantiles = actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)
        chosen_action_qvals = th.gather(mac_out[:,:-1], dim=3, index=actions_for_quantiles).squeeze(3)  # Remove the action dim
        del actions_for_quantiles
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, self.args.n_agents, self.n_quantiles)
        if self.args.mixer == "dplex":
            assert mac_out.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            x_mac_out = mac_out.clone().detach()
            x_mac_out[avail_actions == 0] = -9999999
            max_action_qvals, max_action_index = x_mac_out[:, :-1].mean(dim=4).max(dim=3)
            max_action_index = max_action_index.detach().unsqueeze(3)
        del actions

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t, forward_type="target")
            assert target_agent_outs.shape == (batch.batch_size * self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_agent_outs = target_agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            assert target_agent_outs.shape == (batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_mac_out.append(target_agent_outs)
        del target_agent_outs
        del _

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        assert target_mac_out.shape == (batch.batch_size, episode_length, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)

        # Mask out unavailable actions
        assert avail_actions.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions)
        target_avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        target_mac_out[target_avail_actions[:,1:] == 0] = -9999999
        avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:,1:].mean(dim=4).max(dim=3, keepdim=True)[1]
            del mac_out_detach
            assert cur_max_actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
            cur_max_actions = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            del cur_max_actions
        else:
            # [0] is for max value; [1] is for argmax
            cur_max_actions = target_mac_out.mean(dim=4).max(dim=3, keepdim=True)[1]
            assert cur_max_actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
            cur_max_actions_ = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions_).squeeze(3)
        del target_mac_out
        assert target_max_qvals.shape == (batch.batch_size, episode_length, self.args.n_agents, self.n_target_quantiles)

        if self.args.mixer == "dplex":
            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.args.n_actions,)).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)

        # Mix
        q_attend_regs = 0
        if self.mixer is not None:
            if self.args.mixer == "dplex":
                assert target_max_qvals.shape == (batch.batch_size, episode_length, self.args.n_agents, self.n_target_quantiles)
                # Policy Q
                ans_chosen, q_attend_regs, head_entropies = \
                    self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False, is_v=True)
                # Additional `actions_onehot`, `max_action_qvals`
                ans_adv, _, _ = self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False, actions=actions_onehot,
                    max_q_i=max_action_qvals, is_v=False)
                # Q_tot in Eq.12 (of QPLEX paper)
                chosen_action_qvals = ans_chosen + ans_adv
                # Target Q
                target_chosen, _, _ = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target=True, is_v=True)
                # Additional `actions_onehot`, `max_action_qvals`
                target_adv, _, _ = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target=True,
                                                actions=cur_max_actions_onehot,
                                                max_q_i=target_max_qvals.mean(dim=3), is_v=False)
                # Q_tot in Eq.12 (of QPLEX paper)
                target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target=True)
                chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False)
            assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
            assert target_max_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_target_quantiles)

        # Calculate 1-step Q-Learning targets
        target_samples = rewards.unsqueeze(3) + \
            (self.args.gamma * (1 - terminated)).unsqueeze(3) * \
            target_max_qvals
        del target_max_qvals
        del rewards
        del terminated
        assert target_samples.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_target_quantiles)

        # Quantile Huber loss
        target_samples = target_samples.unsqueeze(3).expand(-1, -1, -1, self.n_quantiles, -1)
        assert target_samples.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
        chosen_action_qvals = chosen_action_qvals.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # u is the signed distance matrix
        u = target_samples.detach() - chosen_action_qvals
        del target_samples
        del chosen_action_qvals
        assert u.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        assert rnd_quantiles.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
        tau = rnd_quantiles.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        assert tau.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # The abs term in quantile huber loss
        abs_weight = th.abs(tau - u.le(0.).float())
        del tau
        assert abs_weight.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # Huber loss
        loss = F.smooth_l1_loss(u, th.zeros(u.shape).cuda(), reduction='none')
        del u
        assert loss.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # Quantile Huber loss
        loss = (abs_weight * loss).mean(dim=4).sum(dim=3)
        del abs_weight
        assert loss.shape == (batch.batch_size, episode_length, n_quantile_groups)

        assert mask.shape == (batch.batch_size, episode_length, 1)
        mask = mask.expand_as(loss)

        # 0-out the targets that came from padded data
        loss = loss * mask

        loss = loss.sum() / mask.sum() + q_attend_regs
        assert loss.shape == ()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
