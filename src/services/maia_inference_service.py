
import os
import sys
import time
import traceback
import queue
from typing import Dict, List, Any, Optional

class MaiaInferenceService:
    """
    Dedicated service for running Maia model inference on GPU.
    Consumes requests from an input queue, batches them, and puts results in an output queue.
    """
    def __init__(self, input_queue, result_dict, settings, batch_size=128, timeout=0.05):
        self.input_queue = input_queue
        self.result_dict = result_dict
        self.settings = settings
        self.batch_size = batch_size
        self.timeout = timeout
        self.model = None
        self.prepared = None
        self.device = "cpu"
        self._running = True

    def initialize(self):
        """Load the model (Process-local)."""
        try:
            print("[MaiaService] Initializing model...")
            from maia2 import model, inference
            import torch
            
            # Check overrides
            overrides = self.settings.get('engine_overrides', {})
            self.device = overrides.get('device')
            if not self.device:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            print(f"[MaiaService] Loading Maia2 on {self.device}...")
            # We use the 'active' model by default or parameterize it?
            # Generator uses 'rapid' by default.
            self.model = model.from_pretrained(type="rapid", device=self.device)
            self.prepared = inference.prepare()
            print("[MaiaService] Model loaded successfully.")
            return True
        except Exception as e:
            print(f"[MaiaService] Failed to load model: {e}")
            traceback.print_exc()
            return False

    def run(self):
        """Main loop."""
        if not self.initialize():
            return

        print("[MaiaService] Ready to process.")
        
        while self._running:
            batch = []
            
            # 1. Collect Batch
            while len(batch) < self.batch_size:
                try:
                    # If we have items, don't wait long. If empty, wait up to timeout.
                    q_timeout = 0.001 if batch else self.timeout
                    item = self.input_queue.get(timeout=q_timeout)
                    
                    if item == "STOP":
                        print("[MaiaService] Stopping...")
                        self._running = False
                        break
                        
                    batch.append(item)
                    
                except queue.Empty:
                    # Timeout reached
                    break
            
            if not batch:
                continue

            # 2. Process Batch
            try:
                self._process_batch(batch)
            except Exception as e:
                print(f"[MaiaService] Error processing batch: {e}")
                for req_id, _, _, _, _ in batch:
                    # Write error result
                    self.result_dict[req_id] = []

    def _process_batch(self, batch):
        from maia2 import inference
        
        for req_id, fen, elo, k, min_prob in batch:
            try:
                move_probs, _ = inference.inference_each(self.model, self.prepared, fen, elo, elo)
                
                if move_probs:
                    inputs = sorted(
                        move_probs.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:k]
                    result = [(m, p) for m, p in inputs if p >= min_prob]
                else:
                    result = []
                
                # Write to shared dict
                self.result_dict[req_id] = result
                
            except Exception as e:
                print(f"[MaiaService] Error on item {req_id}: {e}")
                self.result_dict[req_id] = []

def service_entry_point(input_queue, result_dict, settings):
    """Entry point for the separate process."""
    service = MaiaInferenceService(input_queue, result_dict, settings)
    service.run()
