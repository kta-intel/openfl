# Copyright 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Static grouped assigner module."""

from openfl.component.assigner.assigner import Assigner


class FLEXAssigner(Assigner):
    """The task assigner maintains a list of tasks.

    This assigner is designed to facilitate interoperability between federated learning frameworks. 
    The expectation is that the OpenFL collaborator is tasked with running the external framework's API. 
    By default, all collaborators will run the same single task, which is `start_client_adapter` to 
    start the external framework's client and begin relaying gRPC messages. 

    Attributes:
        task_groups* (list of object): Task groups to assign.
    """

    def __init__(self, task_groups=None, **kwargs):
        """Initializes the FLEXAssigner.

        Args:
            task_groups (list of object): Task groups to assign.
            **kwargs: Additional keyword arguments.
        """
        self.task_groups = task_groups
        super().__init__(**kwargs)

    def define_task_assignments(self):
        """Define task assignments for each round and collaborator.

        This method uses the assigner function to assign tasks to
        collaborators for each OpenFL round.
        """
        if self.task_groups is None:
            self.task_groups = [{"name": "default", "tasks": ['start_client_adapter'], "collaborators": self.authorized_cols}]

        for group in self.task_groups:
            if "tasks" not in group or not group["tasks"]:
                group["tasks"] = ['start_client_adapter']
            if "collaborators" not in group or not group["collaborators"]:
                group["collaborators"] = self.authorized_cols

            # Check if any task other than 'start_client_adapter' is present
            for task in group["tasks"]:
                if task != 'start_client_adapter':
                    raise ValueError(f"Unsupported task '{task}' found. FLEXAssigner only supports 'start_client_adapter'.")

        # Start by finding all of the tasks in all specified groups
        self.all_tasks_in_groups = list(
            {task for group in self.task_groups for task in group["tasks"]}
        )

        # Initialize the map of collaborators for a given task on a given round
        for task in self.all_tasks_in_groups:
            self.collaborators_for_task[task] = {i: [] for i in range(self.rounds)}

        for group in self.task_groups:
            group_col_list = group["collaborators"]
            self.task_group_collaborators[group["name"]] = group_col_list
            for col in group_col_list:
                # For now, we assume that collaborators have the same tasks for
                # every round
                self.collaborator_tasks[col] = {i: group["tasks"] for i in range(self.rounds)}
            # Now populate reverse lookup of tasks->group
            for task in group["tasks"]:
                for round_ in range(self.rounds):
                    # This should append the list of collaborators performing
                    # that task
                    self.collaborators_for_task[task][round_] += group_col_list

    def get_tasks_for_collaborator(self, collaborator_name, round_number):
        """Get tasks for a specific collaborator in a specific round.

        Args:
            collaborator_name (str): Name of the collaborator.
            round_number (int): Round number.

        Returns:
            list: List of tasks for the collaborator in the specified round.
        """
        return self.collaborator_tasks[collaborator_name][round_number]

    def get_collaborators_for_task(self, task_name, round_number):
        """Get collaborators for a specific task in a specific round.

        Args:
            task_name (str): Name of the task.
            round_number (int): Round number.

        Returns:
            list: List of collaborators for the task in the specified round.
        """
        return self.collaborators_for_task[task_name][round_number]