import os
import json
import zlib
import pickle
import lmdb
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, TypeVar, Type, List

# Note: datasets import removed as it's not used in this module

# Type variable for generic type hinting
T = TypeVar('T')

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """
    Configuration for the LMDBStorage class.
    
    This dataclass centralizes all initialization parameters for the
    LMDBStorage class, making the class flexible and easy to configure.

    Attributes:
        db_path (str): Path to the LMDB environment directory. This is where the
                       database files will be stored.
        compress (bool): If True, stored values will be compressed using zlib.
                         This is useful for reducing disk space, especially
                         for text-heavy data.
        compression_level (int): The zlib compression level, from 0 to 9. A level
                                 of 9 provides maximum compression but is slower,
                                 while a lower number is faster with less compression.
    """
    db_path: str = "lmdb_data"
    compress: bool = True
    compression_level: int = 6

class LMDBStorage:
    """
    A class to handle key-value storage using LMDB.

    This class provides a simple and robust API for storing, retrieving, deleting,
    and managing key-value pairs. It uses `pickle` for robust serialization of
    any Python object, allowing you to store complex data structures, and `zlib`
    for optional data compression.
    """
    
    def __init__(self, config: Config = Config()):
        """
        Initializes the LMDBStorage with an LMDB environment.
        
        This method sets up the database connection and configures it based on
        the provided `Config` object.

        Args:
            config (Config): An instance of the `Config` dataclass containing
                             all necessary configuration parameters.
                             
        Raises:
            ValueError: If config is None or invalid
            RuntimeError: If LMDB initialization fails
        """
        if config is None:
            raise ValueError("Config cannot be None")
        if not isinstance(config, Config):
            raise ValueError("Config must be an instance of Config class")
            
        self.config = config
        self.db_path = config.db_path
        self.compress = config.compress
        # Ensure compression level is within valid range [0-9]
        self.compression_level = min(9, max(0, config.compression_level))
        self.env: Optional[lmdb.Environment] = None
        self._setup_db(self.config)
    
    def _setup_db(self, config: Config) -> None:
        """
        Sets up the LMDB environment and main database handle.
        
        This is an internal helper method that handles the low-level details
        of opening or creating the LMDB environment and its primary database.
        
        Args:
            config (Config): The configuration object to use for setup.

        Raises:
            RuntimeError: If the LMDB environment fails to initialize.
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(config.db_path, exist_ok=True)
            # lmdb.open() creates the environment if it doesn't exist.
            self.env = lmdb.open(config.db_path, max_dbs=1)
            # Open the main database handle
            self.db = self.env.open_db()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LMDB: {str(e)}")
            
    def _serialize(self, value: Any) -> bytes:
        """
        Serializes a Python object to bytes using pickle with optional compression.
        
        This method uses the `pickle` module, which can serialize nearly any
        Python object, making this storage solution highly flexible.

        Args:
            value (Any): The Python object to serialize.
            
        Returns:
            bytes: The serialized and optionally compressed byte representation
                   of the input object.
                   
        Raises:
            pickle.PicklingError: If the object cannot be pickled.
        """
        try:
            # Use pickle for robust serialization of any Python object
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except pickle.PicklingError as e:
            logger.error(f"Error pickling object: {e}")
            raise
        
        # Apply compression if enabled
        if self.compress:
            return zlib.compress(data, level=self.compression_level)
        return data
        
    def _deserialize(self, data: bytes) -> Any:
        """
        Deserializes bytes back to the original Python object, with optional decompression.
        
        This is the inverse operation of `_serialize`. It handles decompression
        and uses `pickle` to reconstruct the original Python object from bytes.

        Args:
            data (bytes): The byte data to deserialize.
            
        Returns:
            Any: The deserialized Python object. Returns `None` if the input data
                 is empty or if deserialization fails.
        """
        if not data:
            return None
            
        # Decompress if data was compressed
        try:
            if self.compress:
                data = zlib.decompress(data)
                
            # Use pickle to load the original Python object
            return pickle.loads(data)
        except (zlib.error, pickle.UnpicklingError, TypeError) as e:
            logger.error(f"Error deserializing data: {e}")
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """
        Stores a key-value pair in the database.

        The value is serialized to bytes before being stored. If compression is
        enabled in the configuration, the serialized data will also be compressed.
        
        Args:
            key (str): The unique string key for the value.
            value (Any): The Python object to store. This can be any serializable
                         object, from simple types to complex custom classes.
            
        Returns:
            bool: True if the operation was successful, False otherwise.
            
        Raises:
            TypeError: If the provided key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string.")
            
        try:
            with self.env.begin(write=True) as txn:
                serialized = self._serialize(value)
                txn.put(key.encode('utf-8'), serialized)
            return True
        except Exception as e:
            logger.error(f"Error storing key '{key}': {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value by its key from the database.

        Args:
            key (str): The string key of the value to retrieve.
            default (Any, optional): The value to return if the key is not found.
                                     Defaults to None.
            
        Returns:
            Any: The stored Python object associated with the key. Returns the
                 `default` value if the key does not exist.
                 
        Raises:
            TypeError: If the provided key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string.")
            
        try:
            with self.env.begin() as txn:
                value_bytes = txn.get(key.encode('utf-8'))
                
                if value_bytes is None:
                    return default
                
                return self._deserialize(value_bytes)
        except Exception as e:
            logger.error(f"Error retrieving key '{key}': {e}")
            return default
            
    def delete(self, key: str) -> bool:
        """
        Deletes a key-value pair from the database.
        
        Args:
            key (str): The key of the item to delete.
            
        Returns:
            bool: True if the key was found and deleted, False otherwise.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string.")
            
        try:
            with self.env.begin(write=True) as txn:
                # delete() returns True if the key was deleted, False otherwise
                result = txn.delete(key.encode('utf-8'))
            if not result:
                logger.warning(f"Key '{key}' not found, no deletion performed.")
            return result
        except Exception as e:
            logger.error(f"Error deleting key '{key}': {e}")
            return False
            
    def count_keys(self) -> int:
        """
        Counts the number of key-value pairs in the database.
        
        Returns:
            int: The total number of keys in the database. Returns 0 on error.
        """
        try:
            with self.env.begin() as txn:
                # Get statistics about the database
                stats = txn.stat()
                return stats['entries']
        except Exception as e:
            logger.error(f"Error counting keys: {e}")
            return 0
            
    def has_key(self, key: str) -> bool:
        """
        Checks if a key exists in the database.
        
        Args:
            key (str): The key to check for existence.
            
        Returns:
            bool: True if the key exists, False otherwise.
            
        Raises:
            TypeError: If the provided key is not a string.
        """
        if not isinstance(key, str):
            raise TypeError("Key must be a string.")
            
        try:
            with self.env.begin() as txn:
                value_bytes = txn.get(key.encode('utf-8'))
                return value_bytes is not None
        except Exception as e:
            logger.error(f"Error checking key existence '{key}': {e}")
            return False

    def get_keys(self) -> List[str]:
        """
        Retrieves all keys from the database.
        
        Returns:
            List[str]: A list of all keys in the database as strings.
        """
        keys = []
        try:
            with self.env.begin() as txn:
                # Use a cursor to iterate through all key-value pairs
                cursor = txn.cursor()
                for key, _ in cursor:
                    keys.append(key.decode('utf-8'))
        except Exception as e:
            logger.error(f"Error retrieving keys: {e}")
        return keys

    def all_items(self) -> Dict[str, Any]:
        """
        Retrieves all key-value pairs from the database as a dictionary.
        
        Returns:
            Dict[str, Any]: A dictionary where keys are strings and values are
                            the deserialized Python objects from the database.
        """
        result = {}
        try:
            with self.env.begin() as txn:
                cursor = txn.cursor()
                for key_bytes, value_bytes in cursor:
                    try:
                        key_str = key_bytes.decode('utf-8')
                        result[key_str] = self._deserialize(value_bytes)
                    except Exception as e:
                        logger.error(f"Error processing item with key {key_bytes}: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error retrieving all items: {e}")
        return result
        
    def from_json(self, file_path: str, clear_existing: bool = False) -> bool:
        """
        Imports data from a JSON file into the LMDB database.
        
        This method is designed to load data from a standard JSON file where
        the top-level structure is a key-value dictionary. All values are
        stored as Python objects using `pickle` serialization.

        Args:
            file_path (str): The path to the JSON file containing the data.
            clear_existing (bool, optional): If True, all existing data in the
                                             database will be cleared before
                                             importing the new data. Defaults to False.
                            
        Returns:
            bool: True if the import was successful, False otherwise.
            
        Raises:
            FileNotFoundError: If the specified JSON file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        from pathlib import Path
        
        # Check if file exists
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
            
        try:
            # Read and parse JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Clear existing data if requested
            if clear_existing:
                self.clear()
                
            # Store data in LMDB
            for key, value in data.items():
                self.put(str(key), value)
                    
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error importing data from JSON: {e}")
            return False
            
    def clear(self) -> bool:
        """
        Clears all data from the database.
        
        This operation uses a fast, low-level LMDB command to empty the database.
        
        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        try:
            with self.env.begin(write=True) as txn:
                # Use a fast method to delete all keys
                txn.drop(self.db, delete=False)
            return True
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return False
            
    def close(self) -> None:
        """Closes the database connection, freeing up system resources."""
        if self.env is not None:
            self.env.close()
            self.env = None
            
    def __enter__(self) -> 'LMDBStorage':
        """
        Enables use of this class with the 'with' statement.
        
        Returns:
            LMDBStorage: The instance of the storage class.
        """
        return self
    
    def __exit__(self, exc_type: Optional[type], exc_val: Optional[Exception], exc_tb: Optional[object]) -> None:
        """
        Ensures the database connection is closed when exiting a 'with' block.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        self.close()
        
    def __len__(self) -> int:
        """
        Returns the number of key-value pairs in the database.
        
        This method allows for use with Python's built-in `len()` function.
        """
        return self.count_keys()
        
    def __iter__(self) -> iter:
        """
        Returns an iterator over the keys in the database.
        
        This method allows for iterating directly over the storage keys, such as
        in a `for` loop.
        """
        return iter(self.get_keys())

# Example usage
if __name__ == "__main__":
    # Define a custom class to demonstrate pickle serialization
    class User:
        def __init__(self, name: str, email: str):
            self.name = name
            self.email = email
            
        def __repr__(self) -> str:
            return f"User(name='{self.name}', email='{self.email}')"

    # --- New Example for a Custom 'Derive' Class ---
    # This demonstrates how to store your own classes
    class BaseDataset:
        def __init__(self, name: str):
            self.name = name

    class DerivedClass(BaseDataset):
        def __init__(self, name: str, version: str, data: dict):
            super().__init__(name)
            self.version = version
            self.data = data
        
        def display_info(self):
            return f"DerivedClass: {self.name} (v{self.version}) with {len(self.data)} items."

    print("--- Storing and Retrieving a Custom Derived Class ---")
    
    # Create an instance of the derived class
    my_derived_object = DerivedClass(
        name="my-custom-dataset",
        version="1.0",
        data={"item1": 123, "item2": 456}
    )
    
    # Create a new storage instance for the custom class example
    with LMDBStorage(config=Config(db_path="lmdb_custom_class")) as custom_class_storage:
        custom_class_storage.clear()
        
        # Store the object directly
        custom_class_storage.put("my_dataset_instance", my_derived_object)
        
        # Retrieve the object from the database
        retrieved_object = custom_class_storage.get("my_dataset_instance")
        
        print("Stored object:", my_derived_object)
        print("Retrieved object:", retrieved_object)
        print("Is retrieved object an instance of DerivedClass?", isinstance(retrieved_object, DerivedClass))
        print("Display info from retrieved object:", retrieved_object.display_info())
        
        # You can also store a list of these objects
        list_of_objects = [
            DerivedClass(name="A", version="1", data={"key": "val"}),
            DerivedClass(name="B", version="2", data={})
        ]
        custom_class_storage.put("list_of_objects", list_of_objects)
        retrieved_list = custom_class_storage.get("list_of_objects")
        print("\nRetrieved list of objects:", retrieved_list)

    print("\nDatabase connection for custom class example closed.")
    print("\n-----------------------------------------------------")

    # --- Original Examples from Previous Versions ---
    # Create a storage instance with default config
    with LMDBStorage() as storage:
        # Clear previous data for a clean run
        storage.clear()
        
        # Store some data, including a custom object
        user_object = User("Jane Doe", "jane.doe@example.com")
        storage.put("user:1:name", "John Doe")
        storage.put("user:1:preferences", {"theme": "dark", "notifications": True})
        storage.put("user:2:object", user_object)
        
        print("Number of keys:", len(storage))
        print("Keys:", storage.get_keys())
        
        # Retrieve data
        print("\n--- Retrieving Data ---")
        print("User 1 name:", storage.get("user:1:name"))
        print("User 1 preferences:", storage.get("user:1:preferences"))
        print("User 2 object:", storage.get("user:2:object"))
        print("Non-existent key:", storage.get("user:99", "Not found"))
        
        # Check for key existence
        print("\n--- Key Existence ---")
        print("Has key 'user:1:name'?", storage.has_key("user:1:name"))
        print("Has key 'user:99'?", storage.has_key("user:99"))
        
        # Retrieve all data as a dict
        print("\n--- All Items ---")
        all_items = storage.all_items()
        for key, value in all_items.items():
            print(f"'{key}': {value}")
            
        # Delete a key
        print("\n--- Deleting 'user:1:name' ---")
        deleted = storage.delete("user:1:name")
        print("Deletion successful:", deleted)
        print("User 1 name after deletion:", storage.get("user:1:name", "Not found"))
        
        print("\n--- Final State ---")
        print("Number of keys:", len(storage))
        print("Remaining items:", storage.all_items())

    print("\nDatabase connection closed. To clear the directory, delete 'my_database_lmdb'.")

    # Example with custom config
    with LMDBStorage(config=Config(db_path="my_custom_db", compress=False)) as custom_storage:
        custom_storage.put("test_key", "This is a non-compressed value.")
        print(f"\nStored value in '{custom_storage.db_path}':", custom_storage.get("test_key"))

