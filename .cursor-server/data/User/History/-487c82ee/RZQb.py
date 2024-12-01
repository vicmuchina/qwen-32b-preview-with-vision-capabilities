# ... existing imports ...
# import bitsandbytes as bnb  # Remove or comment out
# ... existing imports ...

import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class InteractiveChat:
    def __init__(self):
        logger.info("Initializing chat interface...")
        print("Setting up model (this may take a few minutes)...")
        self.pipeline = MultimodalPipeline()
        print("Setup complete! Ready for chat.")
        # ... rest of the code ... 