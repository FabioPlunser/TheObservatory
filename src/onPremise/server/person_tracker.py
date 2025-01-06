import multiprocessing as mp
import cv2
import numpy as np
import logging
import time
import torch
import asyncio
import uuid
import aiohttp

from bytetrack.byte_tracker import BYTETracker
from typing import Dict, List, Optional, Tuple
from nats_client import SharedNatsClient, Commands
from logging_config import setup_logger

setup_logger()
logger = logging.getLogger("PersonTracker")
