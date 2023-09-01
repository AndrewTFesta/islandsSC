"""
@title

@description

"""
import json
import logging
import pickle
import threading
import time
import uuid
from pathlib import Path

from tqdm import tqdm


class MAIsland:

    def __init__(
            self, agent_populations, evolving_agent_names, env, optimizer, max_iters, save_dir, migrate_every=1,
            name=None, track_progress=False, logger=None
    ):
        """
        neighbors are a list of "neighboring" island where an island is able to migrate populations to its neighbors
        able to know which agents to migrate to which island by looking at the agents being evolved on the current island and the neighbor island
        note that there is no restriction that any given island may be the only island evolving a certain "type" of agent
          or that "neighboring" must be symmetric

        :param agent_populations:
        :param evolving_agent_names:
        :param env:
        :param optimizer:
        :param max_iters:
        :param save_dir:
        :param migrate_every:
        :param name:
        :param track_progress:
        :param logger:
        """
        if name is None:
            name = '_'.join([str(agent_type) for agent_type in evolving_agent_names])
            name = f'MAIsland__[{name}]'
        if logger is None:
            logger = logging.getLogger()

        # todo  mark where migrations happen
        self.logger = logger
        self.running = False
        self.track_progress = track_progress
        self.optimizer_func = optimizer
        self.env = env

        self.neighbors: list[MAIsland] = []
        self.agent_populations = agent_populations
        self.name = f'{name}_{str(uuid.uuid4())[-4:]}'
        self.evolving_agent_names = evolving_agent_names
        self.migrate_every = migrate_every
        self.since_last_migration = 0
        self.max_iters = max_iters
        self.save_dir = save_dir
        self.times_fname = Path(self.save_dir, 'opt_times.json')
        self.migrated_from_neighbors = {}
        self.num_migrations = 0
        self.total_gens_run = 0
        self.opt_times = None
        self.final_pops = None
        self.top_inds = None
        return

    def __repr__(self):
        return f'{self.name}'

    def __getstate__(self):
        island_state = {
            'neighbors': self.neighbors, 'agent_populations': self.agent_populations, 'name': self.name,
            'evolving_agent_names': self.evolving_agent_names, 'migrate_every': self.migrate_every,
            'since_last_migration': self.since_last_migration, 'max_iters': self.max_iters, 'save_dir': self.save_dir,
            'times_fname': self.times_fname, 'migrated_from_neighbors': self.migrated_from_neighbors, 'num_migrations': self.num_migrations,
            'total_gens_run': self.total_gens_run, 'final_pops': self.final_pops, 'top_inds': self.top_inds
        }
        return island_state

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        return

    def sort_population(self, agent_type):
        sorted_pop = sorted(self.agent_populations[agent_type], key=lambda x: x.fitness, reverse=True)
        self.agent_populations[agent_type] = sorted_pop
        return

    def optimize(self):
        logging.debug(msg=f'Running island optimizer on thread: {threading.get_native_id()}')
        # run the optimize function to completion (as defined by the optimize function)
        self.total_gens_run = 0
        self.opt_times = {'times': [], 'num_gens': []}
        self.final_pops = None
        self.top_inds = None
        pbar = tqdm(total=self.max_iters, desc=f'{self.name}') if self.track_progress else None
        self.num_migrations = 0
        self.running = True
        self.save_island()
        while self.running and self.total_gens_run < self.max_iters:
            if self.agents_migrated():
                # todo  add a minimum number of gens that must be run after incorporating a migration before a new migration can be incorporated
                self.incorporate_migrations()

            # todo  check calculating remaining gens
            #       seems to oboe on every migration
            opt_start = time.process_time()
            self.final_pops, self.top_inds, gens_run = self.optimizer_func(
                agent_policies=self.agent_populations, env=self.env, starting_gen=self.total_gens_run, max_iters=self.total_gens_run + self.migrate_every,
                completion_criteria=self.interrupt_criteria, track_progress=pbar, experiment_dir=self.save_dir
            )
            opt_end = time.process_time()
            opt_time = opt_end - opt_start
            # todo  also save number of generations run since last
            self.opt_times['times'].append(opt_time)
            self.opt_times['num_gens'].append(gens_run)
            self.total_gens_run += gens_run
            self.since_last_migration += gens_run

            total_time = sum(self.opt_times['times'])
            time_per_gen_opt = opt_time / (gens_run + 0.001)
            time_per_gen_overall = total_time / self.total_gens_run

            remaining_gens = self.max_iters - self.total_gens_run
            remaining_time = remaining_gens * time_per_gen_overall

            max_fitnesses = {agent_type: max([each_policy.fitness for each_policy in policies]) for agent_type, policies in self.top_inds.items()}
            info_message = f'Island {self.name}:{self.total_gens_run}/{self.max_iters}:{remaining_time=}'
            debug_message = (f'Island {self.name}:{total_time=}:{time_per_gen_overall=}:{remaining_gens=}:'
                             f'{gens_run=}:{opt_time=}:{time_per_gen_opt=}:{max_fitnesses}')
            logging.debug(msg=info_message)
            logging.debug(msg=debug_message)
            with open(self.times_fname, 'w+') as times_file:
                json.dump(self.opt_times, times_file)

            # at the end of every optimization loop (when the optimizer finishes),
            # migrate the champion from all learning population to every neighbor island
            if self.since_last_migration >= self.migrate_every:
                # this guards against an early migration when we stop the optimizer in order
                # to incorporate a new population that has been migrated from another island
                self.send_populations()
                self.num_migrations += 1
                self.since_last_migration = 0
            self.save_island()
        if isinstance(pbar, tqdm):
            pbar.close()
        self.running = False
        return self.final_pops, self.top_inds, self.opt_times

    def interrupt_criteria(self):
        completion_criteria = (
            self.agents_migrated(),
            self.migration_criteria()
        )
        completed = any(completion_criteria)
        return completed

    def agents_migrated(self):
        agents_migrated = [len(policies) > 0 for agent_type, policies in self.migrated_from_neighbors.items()]
        agents_migrated = any(agents_migrated)
        return agents_migrated

    @staticmethod
    def migration_criteria():
        # todo  this could also include mcc criteria (criteria to migrate pops to another island)
        return False

    def incorporate_migrations(self):
        for agent_type, population in self.migrated_from_neighbors.items():
            # determine how many old policies must be kept to satisfy env requirements
            # add the new policies with the current population
            num_agents = self.env.num_agent_types(agent_type)
            num_agents -= max(len(population), 0)
            self.sort_population(agent_type)
            new_agents = self.agent_populations[agent_type]
            if agent_type not in self.evolving_agent_names:
                # keep the top N previous policies if this island is not evolving this type of agent
                # this ensures that if integrating a new population, the current populations being
                # evolved do not get thrown out and can still have an influence on the learning on this island
                new_agents = new_agents[:num_agents]
            new_agents.extend(population)

            logging.debug(f'Incorporate: {time.time()}: {agent_type}: {len(new_agents)}: Island {self.name}')
            self.agent_populations[agent_type] = new_agents

        # reset view of migrated agents so that the same populations are not migrated repeatedly
        self.migrated_from_neighbors = {}
        return

    def receive_population(self, pop_id, population, from_neighbor):
        # todo  replace with dict.get
        if pop_id not in self.migrated_from_neighbors:
            self.migrated_from_neighbors[pop_id] = []

        self.migrated_from_neighbors[pop_id].extend(population)
        return

    def send_populations(self):
        for each_neighbor in self.neighbors:
            for agent_type, population in self.agent_populations.items():
                if agent_type in self.evolving_agent_names:
                    num_agents = each_neighbor.env.num_agent_types(agent_type)
                    self.sort_population(agent_type)
                    top_agents = self.agent_populations[agent_type]
                    top_agents = top_agents[:num_agents]

                    logging.debug(f'Migration: {time.time()}: {len(top_agents)} agents: {self.name} -> {each_neighbor.name}')
                    each_neighbor.receive_population(agent_type, top_agents, self)
        return

    def close(self):
        self.env.close()
        return

    def save_island(self, save_dir=None, tag=''):
        save_dir = self.save_dir if save_dir is None else save_dir
        tag = f'_{tag}' if tag != '' else tag

        save_path = Path(save_dir, f'island_{self.name}{tag}.pkl')
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as save_file:
            pickle.dump(self, save_file, pickle.HIGHEST_PROTOCOL)
        return save_path

    @staticmethod
    def load_environment(island_path):
        with open(island_path, 'rb') as load_file:
            island = pickle.load(load_file)
        return island
