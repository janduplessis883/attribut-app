import streamlit as st
from supabase import create_client

# Supabase setup
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase = create_client(supabase_url, supabase_key)

st.title("Attribut-AI User Signup")

response = supabase.auth.sign_up(
    {"email": "drjanduplessis@icloud.com", "password": "Password"}
)
