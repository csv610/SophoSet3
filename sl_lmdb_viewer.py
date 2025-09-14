import os
import streamlit as st
from PIL import Image
import io
import numpy as np
from typing import Any, List, Dict, Optional
from lmdb_storage import LMDBStorage, Config

# Set page config
st.set_page_config(page_title="LMDB Viewer", layout="wide")

def display_value(value: Any) -> Any:
    """Display value based on its type"""
    if value is None:
        return "Value is None"
    
    if isinstance(value, bytes):
        # Try to display as image
        try:
            img = Image.open(io.BytesIO(value))
            st.image(img, caption="Image from bytes")
            return
        except:
            pass
        
        # Try to display as text
        try:
            return value.decode('utf-8')
        except:
            return f"Binary data (length: {len(value)} bytes)"
    
    elif isinstance(value, (str, int, float, bool)):
        return value
    
    elif isinstance(value, (list, dict, tuple, set)):
        return value
    
    elif isinstance(value, np.ndarray):
        st.write(f"Numpy array shape: {value.shape}")
        st.write(value)
        return
    
    # Try to get a string representation of custom objects
    try:
        return str(value)
    except:
        return f"[Object of type {type(value).__name__}]"

def get_all_keys(storage: LMDBStorage) -> List[str]:
    """Get all keys from LMDB storage"""
    keys = []
    with storage.env.begin() as txn:
        cursor = txn.cursor()
        for key, _ in cursor:
            try:
                keys.append(key.decode('utf-8'))
            except UnicodeDecodeError:
                # Handle non-UTF-8 keys
                keys.append(str(key))
    return keys

def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='LMDB Database Viewer')
    parser.add_argument('db_path', type=str, nargs='?', default=os.path.join(os.getcwd(), "lmdb_data"),
                      help='Path to the LMDB database directory')
    return parser.parse_known_args()

def main():
    # Parse command line arguments
    args, _ = parse_args()
    
    st.title("LMDB Database Viewer")
    
    # Sidebar for database information
    st.sidebar.header("LMDB Configuration")
    
    # Use the provided path
    lmdb_path = args.db_path
    
    # Display the current path
    st.sidebar.write(f"**Database Directory:**")
    st.sidebar.code(lmdb_path, language="text")
    
    if not os.path.isdir(lmdb_path):
        st.error(f"Error: Directory does not exist: {lmdb_path}")
        st.stop()
    
    # Add items per page selection
    items_per_page = st.sidebar.slider(
        "Items per page", 
        min_value=10, 
        max_value=100, 
        value=20, 
        step=5,
        help="Number of items to display per page"
    )
    
    # Database configuration section
    st.sidebar.markdown("### Database Settings")
    st.sidebar.info("This is a read-only viewer. Database settings cannot be modified.")
    
    try:
        # Initialize LMDBStorage with just the database path
        storage = LMDBStorage(Config(db_path=lmdb_path))
        
        # Get all keys
        keys = get_all_keys(storage)
        
        if not keys:
            st.warning("The database is empty")
            return
            
        # Display database stats
        with st.expander("Database Info"):
            st.write(f"Database path: {os.path.abspath(lmdb_path)}")
            st.write(f"Total entries: {len(keys)}")
            st.write(f"Sample keys: {keys[:5] + ['...'] if len(keys) > 5 else keys}")
        
        # Pagination
        total_pages = max(1, (len(keys) + items_per_page - 1) // items_per_page)
        page = st.sidebar.number_input(
            "Page", 
            min_value=1, 
            max_value=total_pages, 
            value=1, 
            step=1,
            help=f"Page 1 of {total_pages}"
        )
        
        # Calculate start and end indices for current page
        start_idx = (page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(keys))
        
        # Display current page info
        st.write(f"Showing items {start_idx + 1} to {end_idx} of {len(keys)}")
        
        # Display keys for current page
        st.subheader("Available Keys")
        for i in range(start_idx, end_idx):
            key = keys[i]
            with st.expander(f"{i+1}. {key}"):
                value = storage.get(key)
                st.write(display_value(value))
                
                # Add a button to view full details
                if st.button(f"View Details", key=f"btn_{i}"):
                    st.session_state['selected_key'] = key
        
        # Display detailed view if a key is selected
        if 'selected_key' in st.session_state and st.session_state['selected_key'] in keys:
            selected_key = st.session_state['selected_key']
            st.subheader(f"Detailed View: {selected_key}")
            value = storage.get(selected_key)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Key", selected_key)
            with col2:
                st.metric("Value Type", type(value).__name__)
            
            st.subheader("Value")
            st.write(display_value(value))
            
            # Add a button to close the detailed view
            if st.button("Close Detailed View"):
                del st.session_state['selected_key']
                st.experimental_rerun()
                
    except Exception as e:
        st.error(f"Error accessing LMDB database: {str(e)}")
    
    # Cleanup
    if 'storage' in locals():
        storage.env.close()

if __name__ == "__main__":
    main()
