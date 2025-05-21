"""
Simple Authentication for Stock Price Prediction App
This provides a basic login interface before allowing access to the main application.
"""
import os
import streamlit as st
import pickle
import hashlib
import pandas as pd
from datetime import datetime

# Configure Streamlit page
if 'page_config_set' not in st.session_state:
    st.set_page_config(
        page_title="Stock Price Predictor - Login",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.session_state.page_config_set = True

# File to store users
USERS_FILE = "./.streamlit/users.pkl"

# Create directory if it doesn't exist
os.makedirs("./.streamlit", exist_ok=True)

# Function to hash passwords
def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

# Function to create empty users file if it doesn't exist
def create_empty_users():
    """Create an empty users dictionary."""
    users = {}
    
    with open(USERS_FILE, "wb") as file:
        pickle.dump(users, file)
    
    return users

# Function to load users
def load_users():
    """Load users from file or create empty users dictionary."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "rb") as file:
            try:
                return pickle.load(file)
            except Exception as e:
                st.error(f"Error loading users: {e}")
                return create_empty_users()
    else:
        return create_empty_users()

# Function to save users
def save_users(users):
    """Save users to file."""
    with open(USERS_FILE, "wb") as file:
        pickle.dump(users, file)

# Function to verify login
def verify_login(username, password):
    """Verify login credentials."""
    users = load_users()
    if username in users and users[username]["password"] == hash_password(password):
        return True, users[username]
    return False, {"name": None}

# Function to add a new user
def add_user(username, password, name, email, role="user"):
    """Add a new user."""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    
    users[username] = {
        "password": hash_password(password),
        "name": name,
        "email": email,
        "role": role,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    save_users(users)
    return True, "User added successfully"

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None

# Login functionality
def main():
    """Main function to handle authentication and app navigation."""
    if not st.session_state.authenticated:
        # Login form
        st.title("Stock Price Predictor")
        
        # Create tabs for login and registration
        login_tab, register_tab = st.tabs(["Login", "Register"])
        
        with login_tab:
            with st.form("login_form"):
                st.subheader("Login")
                username = st.text_input("Username", key="login_username")
                password = st.text_input("Password", type="password", key="login_password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    verified, user = verify_login(username, password)
                    if verified and user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        # Store user info in session state for app.py
                        st.session_state.username = username
                        st.session_state.user_name = user.get('name', 'User')
                        st.session_state.user_email = user.get('email', '')
                        st.session_state.user_role = user.get('role', 'user')
                        # Initialize the page to dashboard by default
                        st.session_state.page = "dashboard"
                        st.success(f"Welcome, {user['name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
        
        with register_tab:
            with st.form("register_form"):
                st.subheader("Create an Account")
                new_username = st.text_input("Username", key="register_username")
                new_password = st.text_input("Password", type="password", key="register_password")
                confirm_password = st.text_input("Confirm Password", type="password")
                new_name = st.text_input("Full Name")
                new_email = st.text_input("Email")
                
                register_submit = st.form_submit_button("Register")
                
                if register_submit:
                    # Validate inputs
                    if not (new_username and new_password and confirm_password and new_name and new_email):
                        st.error("All fields are required")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif "@" not in new_email or "." not in new_email:
                        st.error("Please enter a valid email address")
                    else:
                        # Check if username already exists
                        users = load_users()
                        if new_username in users:
                            st.error("Username already exists. Please choose another one.")
                        else:
                            # Create new user (default role is 'user')
                            success, message = add_user(new_username, new_password, new_name, new_email, "user")
                            if success:
                                st.success("Registration successful! You can now log in.")
                            else:
                                st.error(message)
    else:
        # User is authenticated
        # Add logout button to sidebar with unique key
        if st.sidebar.button("Logout", key="auth_logout_button"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
        
        # Show user information
        st.sidebar.success(f"Logged in as: {st.session_state.user['name']}")
        
        # Admin panel if user is admin
        if st.session_state.user["role"] == "admin":
            if st.sidebar.checkbox("Admin Panel"):
                st.title("User Management")
                
                # Add user form
                with st.expander("Add New User"):
                    with st.form("add_user_form"):
                        new_username = st.text_input("Username")
                        new_password = st.text_input("Password", type="password")
                        new_name = st.text_input("Name")
                        new_email = st.text_input("Email")
                        new_role = st.selectbox("Role", ["user", "admin"])
                        
                        add_user_submit = st.form_submit_button("Add User")
                        
                        if add_user_submit:
                            if new_username and new_password and new_name and new_email:
                                success, message = add_user(new_username, new_password, new_name, new_email, new_role)
                                if success:
                                    st.success(message)
                                else:
                                    st.error(message)
                            else:
                                st.warning("All fields are required")
                
                # List all users
                users = load_users()
                st.subheader("Registered Users")
                
                # Create a DataFrame for better display
                user_data = []
                for username, user_info in users.items():
                    user_data.append({
                        "Username": username,
                        "Name": user_info["name"],
                        "Email": user_info["email"],
                        "Role": user_info["role"],
                        "Created At": user_info["created_at"]
                    })
                
                if user_data:
                    st.dataframe(pd.DataFrame(user_data))
                else:
                    st.info("No users found")
            else:
                # Properly import and run the main app
                import app
                # Make sure app module is fully loaded 
                from app import main
                main()
        else:
            # Regular user - run the main app
            import app
            # Properly import the main function from app module
            from app import main
            main()

if __name__ == "__main__":
    main()