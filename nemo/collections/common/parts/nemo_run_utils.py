# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import lru_cache

from nemo_run.core.tunnel import LocalTunnel, SSHTunnel
from omegaconf import DictConfig, OmegaConf

from nemo.collections.common.parts.skills_utils import add_task, check_if_mounted, get_mounts_from_config, run_exp
from nemo.utils import logging

__all__ = [
    "add_task",
    "check_if_mounted",
    "get_mounts_from_config",
    "run_exp",
    "get_tunnel",
    "add_mount_path",
    "create_remote_directory",
    "create_remote_config",
    "check_remote_mount_directories",
    "get_unmounted_filepath",
    "get_mounted_filepath",
]


@lru_cache(maxsize=2)
def get_tunnel(**ssh_tunnel):
    """
    Establishing ssh tunnel
    """
    return SSHTunnel(**ssh_tunnel)


def add_mount_path(mount_source: str, mount_dest: str, cluster_config):
    """
    Add a mount path to the cluster config.

    Args:
        mount_source: The source filepath on the local/remote machine.
        mount_dest: The destination filepath on the remote/local machine. Must be an absolute path.
        cluster_config: The cluster config dictionary.
    """

    # Check if the cluster config is provided
    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    # Check if the mounts key is present in the cluster config
    if 'mounts' in cluster_config:
        # Resolve the environment variables for the mount source and mount destination
        original_mounts = get_mounts_from_config(cluster_config)

        added_mount = False
        for mount_path in original_mounts:
            source, destination = mount_path.split(':')

            # Check if the mount path already exists in the cluster config
            if source == mount_source and destination == mount_dest:
                return

        # Add the mount path to the cluster config if it does not already exist
        if not added_mount:
            cluster_config['mounts'].append(f"{mount_source}:{mount_dest}")
            logging.info(f"Added mount path: `{mount_source}:{mount_dest}`")

    else:
        # Don't add a new mount path if the mounts key is not present in the cluster config
        raise ValueError("No mounts found in cluster config, can only add to existing mount list.")


def create_remote_directory(directory: str | list, cluster_config: dict):
    """
    Create a remote directory on the cluster using the cluster config.

    **Note**: The ssh tunnel config must be provided in the cluster config for remote directory creation.

    Args:
        directory: The directory path to be created on the remote cluster. Can be a single directory path or a list
            of directory paths.
        cluster_config: The cluster config dictionary.
    """

    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    # Check if the directory is a string or a list
    if isinstance(directory, str):
        directory = [directory]

    # Check if the executor is local
    if cluster_config.get('executor') == 'local':
        tunnel = LocalTunnel(job_dir=directory[0])  # temp job dir, unused
        for dir_path in directory:
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            logging.info(f"Created directory: {dir_path} in local filesystem.")

        # Dont cleanup, cache the tunnel
        # tunnel.cleanup()

    # Check if the executor is slurm
    elif cluster_config.get('executor') == 'slurm':
        # Check if the ssh tunnel config is provided in the cluster config
        ssh_tunnel_config = cluster_config.get('ssh_tunnel', None)
        if ssh_tunnel_config is None:
            raise ValueError("`ssh_tunnel` sub-config is not provided in cluster_config.")

        # Check for pre-existing job_dir in the ssh_tunnel_config
        if 'job_dir' not in ssh_tunnel_config:
            ssh_tunnel_config['job_dir'] = directory[0]

        # Create the remote directory on the cluster
        tunnel = get_tunnel(**cluster_config['ssh_tunnel'])
        for dir_path in directory:
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            logging.info(f"Created directory: {dir_path} on remote cluster.")

        # Dont cleanup, cache the tunnel
        # tunnel.cleanup()

    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def create_remote_config(config: dict | DictConfig, config_name: str, config_directory: str, cluster_config: dict):
    """
    Utility to write a remote config file on the cluster using the cluster config.

    Args:
        config: The config dictionary to be written to the file. Can be OmegaConf DictConfig or a dictionary.
        config_name: The name of the config file to be created.
        config_directory: The directory path where the config file will be created on the remote machine.
            Can be a single directory path or a list of directory paths to copy the config file to.
        cluster_config: The cluster config dictionary.
    """
    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    # Check if the config_name is a string and ends with .yaml
    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"

    # Check if the config_directory is a string or a list
    if isinstance(config_directory, str):
        config_directory = [config_directory]

    # Cast a normal dict to OmeagConf DictConfig
    if isinstance(config, dict):
        config = OmegaConf.create(config)

    # Check if the executor is local
    if cluster_config.get('executor') == 'local':
        tunnel = LocalTunnel(job_dir=config_directory[0])

        # Create the config file on the local filesystem
        for dir_path in config_directory:
            config_filepath = os.path.join(dir_path, config_name)
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            tunnel.run(f"touch {config_filepath}", hide=False, warn=True)
            tunnel.run(f"echo '{OmegaConf.to_yaml(config)}' > {config_filepath}", hide=False, warn=True)
            logging.info(f"Created config file: {dir_path} in local filesystem.")

        # Dont cleanup, cache the tunnel
        # tunnel.cleanup()

    # Check if the executor is slurm
    elif cluster_config.get('executor') == 'slurm':
        # Check if the ssh tunnel config is provided in the cluster config
        ssh_tunnel_config = cluster_config.get('ssh_tunnel', None)
        if ssh_tunnel_config is None:
            raise ValueError("`ssh_tunnel` sub-config is not provided in cluster_config.")

        # Check for pre-existing job_dir in the ssh_tunnel_config
        if 'job_dir' not in ssh_tunnel_config:
            ssh_tunnel_config['job_dir'] = config_directory[0]

        tunnel = get_tunnel(**cluster_config['ssh_tunnel'])

        # Create the config file on the remote cluster
        for dir_path in config_directory:
            config_filepath = os.path.join(dir_path, config_name)
            tunnel.run(f'mkdir -p {dir_path}', hide=False, warn=True)
            tunnel.run(f"touch {config_filepath}", hide=False, warn=True)
            tunnel.run(f"echo '{OmegaConf.to_yaml(config)}' > {config_filepath}", hide=False, warn=True)
            logging.info(f"Created config file: {dir_path} on remote cluster.")

        # Dont cleanup, cache the tunnel
        # tunnel.cleanup()

    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def check_remote_mount_directories(directories: str | list, cluster_config: dict, exit_on_failure: bool = True):
    """
    Check if files and directories at the source location exist for later mounting on the cluster.

    Args:
        directories: The directory path to be checked on the local/remote machine. Can be a single directory
            path or a list. Can be either a file or a directory.
        cluster_config: The cluster config dictionary.
        exit_on_failure: If True, will raise an exception if the directories do not exist at the source location.
    """

    # Check if the cluster config is provided
    if cluster_config is None:
        raise ValueError("Cluster config is not provided.")

    # Check if the directories is a string or a list
    if isinstance(directories, str):
        directories = [directories]

    # Check if the executor is local
    if cluster_config.get('executor') == 'local':
        tunnel = LocalTunnel(job_dir=None)

        # Check if the directories exist at the source location for mounting
        missing_source_locations = []
        for directory in directories:
            result = tunnel.run(f'test -e {directory} && echo "Directory Exists"', hide=True, warn=True)

            if "Directory Exists" not in result.stdout:
                missing_source_locations.append(directory)

        # Dont cleanup, cache the tunnel
        # tunnel.cleanup()

        # Raise an exception if the directories do not exist at the source location
        if len(missing_source_locations) > 0 and exit_on_failure:
            missing_source_locations = [
                f"{loc} DOES NOT exist at source destination" for loc in missing_source_locations
            ]
            missing_source_locations = "\n".join(missing_source_locations)
            raise FileNotFoundError(
                f"Some files or directories do not exist at the source location for mounting !!\n\n"
                f"{missing_source_locations}"
            )

    # Check if the executor is slurm
    elif cluster_config.get('executor') == 'slurm':
        # Check if the ssh tunnel config is provided in the cluster config
        ssh_tunnel_config = cluster_config.get('ssh_tunnel', None)
        if ssh_tunnel_config is None:
            raise ValueError("`ssh_tunnel` sub-config is not provided in cluster_config.")

        # Check for pre-existing job_dir in the ssh_tunnel_config
        if 'job_dir' not in ssh_tunnel_config:
            ssh_tunnel_config['job_dir'] = os.getcwd()

        tunnel = get_tunnel(**cluster_config['ssh_tunnel'])
        missing_source_locations = []

        # Check if the directories exist at the source location for mounting
        for directory in directories:
            result = tunnel.run(f'test -e {directory} && echo "Directory Exists"', hide=True, warn=True)

            if "Directory Exists" not in result.stdout:
                missing_source_locations.append(directory)

        # Dont cleanup, cache the tunnel
        # tunnel.cleanup()

        # Raise an exception if the directories do not exist at the source location
        if len(missing_source_locations) > 0 and exit_on_failure:
            missing_source_locations = [
                f"{loc} DOES NOT exist at source destination" for loc in missing_source_locations
            ]
            missing_source_locations = "\n".join(missing_source_locations)
            raise FileNotFoundError(
                f"Some files or directories do not exist at the source location for mounting !!\n\n"
                f"{missing_source_locations}"
            )

    else:
        raise ValueError(f"Unsupported executor: {cluster_config.get('executor')}")


def get_unmounted_filepath(cluster_config: dict, filepath: str):
    """
    Resolve the mounted filepath using the cluster config to merge the mount source path to the filepath.
    Raises an exception if the mount path is not found for the file path.

    Args:
        cluster_config: The cluster config dictionary.
        filepath: The filepath to be unmounted using the cluster config.

    Returns:
        str: unmounted filepath
    """
    # Find which mount path matches the filepaths prefix
    mount_path = None
    for mount in cluster_config['mounts']:
        mount_source, mount_dest = mount.split(':')
        if filepath.startswith(mount_dest):
            mount_path = mount
            break

    if mount_path is None:
        raise ValueError(
            f"Could not find a mount path for the file path `{filepath}`. Below paths are mounted: \n"
            f"{cluster_config['mounts']}"
        )

    # replace the mount destination inside the filepath with the mount source
    mount_source, mount_dest = mount_path.split(':')
    filepath = mount_source + filepath[len(mount_dest) :]  # replace the mount destination with the mount source

    return filepath


def get_mounted_filepath(cluster_config: dict, filepath: str):
    """
    Resolve the mounted filepath using the cluster config to merge the mount destination path to the filepath.
    Raises an exception if the mount path is not found for the file path.

    Args:
        cluster_config: The cluster config dictionary.
        filepath: The filepath to be mounted using the cluster config.

    Returns:
        str: mounted filepath
    """
    # Find which mount path matches the filepaths prefix
    mount_path = None
    for mount in cluster_config['mounts']:
        mount_source, mount_dest = mount.split(':')
        if filepath.startswith(mount_source):
            mount_path = mount
            break

    if mount_path is None:
        raise ValueError(
            f"Could not find a mount path for the file path `{filepath}`. Below paths are mounted: \n"
            f"{cluster_config['mounts']}"
        )

    # replace the mount destination inside the filepath with the mount source
    mount_source, mount_dest = mount_path.split(':')
    filepath = mount_dest + filepath[len(mount_source) :]  # replace the mount destination with the mount source

    return filepath
