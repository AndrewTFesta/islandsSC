"""
@title

@description

"""
import logging
import threading

from island_influence.learn.island.MAIsland import MAIsland


class ThreadIsland(MAIsland):

    def __init__(self, agent_populations, evolving_agent_names, env, optimizer, max_iters, save_dir, migrate_every=1,
                 name=None, track_progress=False, logger=None):
        super().__init__(agent_populations, evolving_agent_names, env, optimizer, max_iters, save_dir, migrate_every=migrate_every,
                         name=name, track_progress=track_progress, logger=logger)

        self.optimize_thread = threading.Thread(target=super().optimize, daemon=True)
        self._update_lock = threading.Lock()
        return

    # def __getstate__(self):
    #     # state = self.__dict__.copy()
    #     return state

    def optimize(self):
        logging.info(msg=f'{self.name}: Starting optimizer')
        self.running = True
        self.optimize_thread.start()
        return

    def stop(self):
        logging.info(msg=f'{self.name}: stopping optimizer')
        self.running = False
        return

    def incorporate_migrations(self):
        with self._update_lock:
            super().incorporate_migrations()
        return

    def receive_population(self, pop_id, population, from_neighbor):
        with self._update_lock:
            super().receive_population(pop_id, population, from_neighbor)
        return

    @staticmethod
    def load_environment(island_path):
        return
