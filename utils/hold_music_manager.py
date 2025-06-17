#!/usr/bin/env python3
"""
Hold Music Manager Utility

This module provides a clean interface for managing hold music processes
during call transfers or waiting periods.
"""

import asyncio
import os
import sys
from loguru import logger


class HoldMusicManager:
    """Manager class for hold music processes."""

    def __init__(self):
        self.process = None
        self._script_path = self._get_script_path()
        self._default_music_path = self._get_default_music_path()

    def _get_script_path(self) -> str:
        """Get the path to the hold music script."""
        current_dir = os.path.dirname(
            os.path.dirname(__file__))  # Go up from utils/
        return os.path.join(current_dir, "assets", "hold_music", "hold_music.py")

    def _get_default_music_path(self) -> str:
        """Get the path to the default hold music file."""
        current_dir = os.path.dirname(
            os.path.dirname(__file__))  # Go up from utils/
        return os.path.join(current_dir, "assets", "hold_music", "hold_music.wav")

    async def start(self, room_url: str, token: str, wav_file_path: str = None) -> bool:
        """
        Start hold music process.

        Args:
            room_url: Daily room URL
            token: Daily room token
            wav_file_path: Path to WAV file (defaults to bundled hold music)

        Returns:
            True if started successfully, False otherwise
        """
        if self.process and self.process.returncode is None:
            logger.warning("Hold music process is already running")
            return True

        # Use default music if no file specified
        music_file = wav_file_path or self._default_music_path

        if not os.path.exists(self._script_path):
            logger.error(f"Hold music script not found: {self._script_path}")
            return False

        if not os.path.exists(music_file):
            logger.error(f"Music file not found: {music_file}")
            return False

        logger.info(f"Starting hold music with file: {music_file}")

        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                self._script_path,
                "-m", room_url,
                "-t", token,
                "-i", music_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            logger.info(
                f"Hold music process started with PID: {self.process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start hold music process: {e}")
            self.process = None
            return False

    async def stop(self, timeout: float = 5.0) -> bool:
        """
        Stop hold music process gracefully.

        Args:
            timeout: Maximum time to wait for graceful shutdown

        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.process or self.process.returncode is not None:
            logger.debug("No hold music process running to stop")
            return True

        logger.info(f"Stopping hold music process PID: {self.process.pid}")

        try:
            self.process.terminate()

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(self.process.wait(), timeout=timeout)
                logger.info("Hold music process stopped gracefully")
                return True

            except asyncio.TimeoutError:
                logger.warning(
                    "Hold music process didn't stop gracefully, forcing kill")
                self.process.kill()
                await self.process.wait()
                logger.info("Hold music process killed")
                return True

        except Exception as e:
            logger.error(f"Error stopping hold music process: {e}")
            return False
        finally:
            self.process = None

    @property
    def is_running(self) -> bool:
        """Check if hold music process is currently running."""
        return self.process is not None and self.process.returncode is None

    @property
    def pid(self) -> int:
        """Get the PID of the running process, or None if not running."""
        return self.process.pid if self.is_running else None


# Convenience functions for backward compatibility and simple usage
async def start_hold_music(room_url: str, token: str, wav_file_path: str = None) -> HoldMusicManager:
    """
    Start hold music and return a manager instance.

    Args:
        room_url: Daily room URL
        token: Daily room token
        wav_file_path: Path to WAV file (defaults to bundled hold music)

    Returns:
        HoldMusicManager instance (even if start failed)

    Raises:
        RuntimeError: If failed to start hold music
    """
    manager = HoldMusicManager()
    success = await manager.start(room_url, token, wav_file_path)

    if not success:
        raise RuntimeError("Failed to start hold music process")

    return manager


async def stop_hold_music(manager: HoldMusicManager) -> bool:
    """
    Stop hold music using a manager instance.

    Args:
        manager: HoldMusicManager instance

    Returns:
        True if stopped successfully, False otherwise
    """
    if not isinstance(manager, HoldMusicManager):
        logger.error("Invalid manager instance provided")
        return False

    return await manager.stop()
