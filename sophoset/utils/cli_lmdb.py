#!/usr/bin/env python3
"""
LMDB Command Line Viewer

A simple CLI tool to inspect LMDB databases.
"""
import os
import sys
import argparse
import json
import pickle
from PIL import Image
import io
import numpy as np
from typing import Any, List, Dict, Optional

from sophoset.utils.lmdb_storage import LMDBStorage, Config

class LMDBViewer:
    def __init__(self, db_path: str):
        """Initialize the LMDB viewer with the given database path."""
        self.db_path = os.path.expanduser(db_path)
        if not os.path.isdir(self.db_path):
            print(f"Error: Directory does not exist: {self.db_path}")
            sys.exit(1)
            
        try:
            self.storage = LMDBStorage(Config(db_path=self.db_path))
        except Exception as e:
            print(f"Error opening LMDB database: {e}")
            sys.exit(1)
    
    def get_all_keys(self) -> List[str]:
        """Get all keys from the LMDB storage."""
        keys = []
        with self.storage.env.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                try:
                    keys.append(key.decode('utf-8'))
                except UnicodeDecodeError:
                    # Handle non-UTF-8 keys
                    keys.append(str(key))
        return sorted(keys)
    
    def get_value(self, key: str) -> Any:
        """Get a value from the LMDB storage by key."""
        try:
            return self.storage.get(key)
        except Exception as e:
            return f"Error retrieving value: {e}"
    
    def display_value(self, value: Any) -> str:
        """Format a value for display in the CLI."""
        if value is None:
            return "None"
        
        if isinstance(value, bytes):
            # Try to detect image data
            try:
                img = Image.open(io.BytesIO(value))
                return f"[Image: {img.format}, size={img.size}, mode={img.mode}]"
            except:
                pass
            
            # Try to decode as text
            try:
                return value.decode('utf-8')
            except:
                return f"[Binary data: {len(value)} bytes]"
        
        if isinstance(value, (str, int, float, bool)):
            return str(value)
            
        if isinstance(value, (list, dict, tuple, set)):
            return json.dumps(value, indent=2, default=str)
            
        if isinstance(value, np.ndarray):
            return f"Numpy array shape: {value.shape}\n{value}"
            
        return str(value)
    
    def interactive_shell(self):
        """Start an interactive shell to explore the LMDB database."""
        print(f"\nLMDB Viewer - {self.db_path}")
        print("Type 'list' to show all keys, 'get <key>' to view a value, or 'exit' to quit")
        
        while True:
            try:
                cmd = input("\n> ").strip().split()
                if not cmd:
                    continue
                    
                if cmd[0].lower() == 'exit':
                    break
                    
                elif cmd[0].lower() == 'list':
                    keys = self.get_all_keys()
                    print(f"\nFound {len(keys)} keys:")
                    for i, key in enumerate(keys, 1):
                        print(f"{i:4d}. {key}")
                
                elif cmd[0].lower() == 'get' and len(cmd) > 1:
                    key = ' '.join(cmd[1:])
                    value = self.get_value(key)
                    print(f"\nKey: {key}")
                    print("-" * 40)
                    print(self.display_value(value))
                
                elif cmd[0].lower() == 'help':
                    print("\nAvailable commands:")
                    print("  list         - List all keys in the database")
                    print("  get <key>    - View the value for a specific key")
                    print("  help         - Show this help message")
                    print("  exit         - Exit the viewer")
                
                else:
                    print("Unknown command. Type 'help' for available commands.")
            
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='LMDB Database Viewer')
    parser.add_argument('-i', '--db_path', nargs='?', default=os.path.join(os.getcwd(), "lmdb_data"),
                       help='Path to the LMDB database directory (default: ./lmdb_data)')
    
    args = parser.parse_args()
    
    viewer = LMDBViewer(args.db_path)
    viewer.interactive_shell()

if __name__ == "__main__":
    main()
