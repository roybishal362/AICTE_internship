# app.py
import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, List, Optional
import datetime
import json
import random
from pathlib import Path
import logging
from database.handler import DatabaseHandler
from chatbot_1 import LLMHandler, UserProfile
import plotly.graph_objects as go
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# Configure page settings
st.set_page_config(
    page_title="RoY - AI Learning Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for futuristic design
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #1a1c23;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00ff88 !important;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Chat containers */
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
        background: linear-gradient(45deg, #2a2d3a, #1a1c23);
        border: 1px solid #3a3f4b;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* User message */
    .user-message {
        background: linear-gradient(45deg, #1e3a8a, #1e40af);
    }
    
    /* Bot message */
    .bot-message {
        background: linear-gradient(45deg, #065f46, #047857);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #00ff88;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(45deg, #00ff88, #00e676);
        color: #1a1c23;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 255, 136, 0.2);
    }
    
    /* Custom card container */
    .custom-card {
        background: linear-gradient(45deg, #2a2d3a, #1a1c23);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #3a3f4b;
        margin-bottom: 1rem;
    }
    
    /* Animations */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .gradient-animate {
        background: linear-gradient(270deg, #00ff88, #00e676, #00ff88);
        background-size: 200% 200%;
        animation: gradient 3s ease infinite;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    if 'db_handler' not in st.session_state:
        st.session_state.db_handler = DatabaseHandler("sqlite:///chatbot.db")
    if 'chatbot_1' not in st.session_state:
        st.session_state.chatbot_1 = LLMHandler("your-api-key")

def load_lottie_animation(url: str) -> dict:
    """Load Lottie animation from URL"""
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

def display_welcome_animation():
    """Display welcome animation using Lottie"""
    animation_url = "https://assets5.lottiefiles.com/packages/lf20_xqbbzilv.json"
    animation_data = load_lottie_animation(animation_url)
    if not animation_data:
        st.warning("Failed to load Lottie animation")
    else:
        st_lottie(animation_data, speed=1, height=200, key="welcome")

def create_progress_chart(progress_data: Dict) -> go.Figure:
    """Create an interactive progress chart"""
    fig = go.Figure()

    # Add progress bars
    fig.add_trace(go.Bar(
        x=list(progress_data.keys()),
        y=list(progress_data.values()),
        marker=dict(
            color='rgba(0, 255, 136, 0.6)',
            line=dict(color='rgba(0, 255, 136, 1)', width=2)
        ),
        name="Progress"
    ))

    # Customize layout
    fig.update_layout(
        title="User Learning Progress",
        xaxis_title="Categories",
        yaxis_title="Completion (%)",
        template="plotly_dark",
        plot_bgcolor="#1a1c23",
        paper_bgcolor="#1a1c23",
        font=dict(color="#00ff88"),
    )

    return fig

def display_chat_interface():
    """Display chat interface"""
    st.header("RoY - Your AI Learning Assistant 🤖")
    display_welcome_animation()

    with st.container():
        user_message = st.text_input("Type your question or message here...", key="user_input")

        if st.button("Send"):
            if user_message.strip():
                # Store user message
                st.session_state.chat_history.append({"sender": "user", "message": user_message})

                # Get bot response
                response = st.session_state.chatbot_1.py.handle_query(
                    session_id=st.session_state.current_session_id,
                    query=user_message
                )
                # Store bot response
                st.session_state.chat_history.append({"sender": "bot", "message": response})
                st.write(f"Response from bot: {response}")
    # Display chat history
    for chat in st.session_state.chat_history:
        st.markdown(f"""
        <div class="chat-message {'user-message' if chat['sender'] == 'user' else 'bot-message'}">
            <p>{chat['message']}</p>
        </div>
        """, unsafe_allow_html=True)

def display_learning_progress():
    """Display user learning progress"""
    st.sidebar.header("Learning Progress")
    progress_data = st.session_state.db_handler.get_learning_progress(st.session_state.user_profile.user_id)
    if progress_data:
        fig = create_progress_chart(progress_data)
        st.sidebar.plotly_chart(fig, use_container_width=True)
    else:
        st.sidebar.info("No progress data available yet. Start learning to track your progress!")

def display_sidebar():
    """Display sidebar for user profile and navigation"""
    st.sidebar.title("Navigation")
    options = ["Home", "Profile", "Learninsg Progress", "Chat"]
    choice = st.sidebar.radio("Go to", options)

    if choice == "Home":
        st.sidebar.success("Welcome to RoY!")
    elif choice == "Profile":
        st.sidebar.info("User profile management coming soon!")
    elif choice == "Learning Progress":
        display_learning_progress()
    elif choice == "Chat":
        display_chat_interface()

# Main function
def main():
    init_session_state()

    st.title("RoY - AI Learning Assistant")
    display_sidebar()

    if st.session_state.user_profile:
        st.success(f"Welcome back, {st.session_state.user_profile.name}!")
    else:
        st.warning("Please log in to access your profile.")

    # Default to chat interface for now
    display_chat_interface()

if __name__ == "__main__":
    main()




# # app.py
# import streamlit as st
# import time
# import json
# from datetime import datetime
# import plotly.graph_objects as go
# from streamlit_lottie import st_lottie
# from streamlit_option_menu import option_menu
# import extra_streamlit_components as stx
# import streamlit.components.v1 as components
# from backend.core.llm_handler import LLMHandler, UserProfile
# import requests

# # Set page configuration
# st.set_page_config(
#     page_title="AI Academy Assistant",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for futuristic design
# def load_css():
#     st.markdown("""
#         <style>
#         /* Main container */
#         .main {
#             background-color: #0e1117;
#             color: #ffffff;
#         }
        
#         /* Futuristic cards */
#         .stCardContainer {
#             border: 1px solid rgba(28, 131, 225, 0.1);
#             border-radius: 15px;
#             padding: 20px;
#             background: linear-gradient(145deg, rgba(14, 17, 23, 0.9), rgba(14, 17, 23, 0.95));
#             backdrop-filter: blur(10px);
#             box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
#         }
        
#         /* Glowing effects */
#         .glow {
#             box-shadow: 0 0 10px #1c83e1,
#                        0 0 20px #1c83e1,
#                        0 0 30px #1c83e1;
#             animation: glow 1.5s ease-in-out infinite alternate;
#         }
        
#         /* Animated button */
#         .stButton>button {
#             background: linear-gradient(45deg, #1c83e1, #00f2fe);
#             border: none;
#             border-radius: 25px;
#             color: white;
#             transition: all 0.3s ease;
#         }
        
#         .stButton>button:hover {
#             transform: translateY(-2px);
#             box-shadow: 0 5px 15px rgba(0, 242, 254, 0.4);
#         }
        
#         /* Chat container */
#         .chat-container {
#             border-radius: 15px;
#             background: rgba(14, 17, 23, 0.95);
#             backdrop-filter: blur(10px);
#             border: 1px solid rgba(28, 131, 225, 0.2);
#             margin: 10px 0;
#             padding: 15px;
#         }
        
#         /* Custom scrollbar */
#         ::-webkit-scrollbar {
#             width: 5px;
#             height: 5px;
#         }
        
#         ::-webkit-scrollbar-track {
#             background: #0e1117;
#         }
        
#         ::-webkit-scrollbar-thumb {
#             background: #1c83e1;
#             border-radius: 10px;
#         }
        
#         /* Progress bars */
#         .stProgress > div > div > div > div {
#             background-color: #1c83e1;
#         }
        
#         /* Metrics */
#         .metric-container {
#             background: rgba(28, 131, 225, 0.1);
#             border-radius: 10px;
#             padding: 10px;
#             margin: 5px;
#             text-align: center;
#         }
        
#         .metric-value {
#             font-size: 24px;
#             font-weight: bold;
#             color: #1c83e1;
#         }
        
#         .metric-label {
#             font-size: 14px;
#             color: #ffffff;
#         }
#         </style>
#     """, unsafe_allow_html=True)

# # Initialize session state
# def init_session_state():
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
#     if 'user_profile' not in st.session_state:
#         st.session_state.user_profile = UserProfile(
#             user_id=str(int(time.time())),
#             experience_level="beginner",
#             interests=[],
#             learning_style="structured"
#         )
#     if 'current_page' not in st.session_state:
#         st.session_state.current_page = "Chat"
#     if 'llm_handler' not in st.session_state:
#         st.session_state.llm_handler = LLMHandler("your-api-key")

# # Custom components
# def render_chat_message(message, is_user=False):
#     with st.container():
#         if is_user:
#             st.markdown(f"""
#                 <div class="chat-container" style="margin-left: 20%;">
#                     <div style="display: flex; justify-content: flex-end;">
#                         <div style="background: #1c83e1; padding: 10px; border-radius: 15px;">
#                             {message}
#                         </div>
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown(f"""
#                 <div class="chat-container" style="margin-right: 20%;">
#                     <div style="display: flex;">
#                         <div style="background: #2d3748; padding: 10px; border-radius: 15px;">
#                             {message}
#                         </div>
#                     </div>
#                 </div>
#             """, unsafe_allow_html=True)

# def render_roadmap(roadmap_data):
#     st.markdown("### Your Learning Roadmap")
    
#     # Timeline visualization
#     timeline_data = []
#     for phase, details in roadmap_data["timeline"].items():
#         timeline_data.append({
#             "Phase": details["name"],
#             "Duration": details["duration"],
#             "Topics": len(details["topics"])
#         })
    
#     fig = go.Figure()
    
#     # Add timeline events
#     for i, phase in enumerate(timeline_data):
#         fig.add_trace(go.Scatter(
#             x=[i],
#             y=[0],
#             mode="markers+text",
#             name=phase["Phase"],
#             text=[phase["Phase"]],
#             textposition="top center",
#             marker=dict(size=20, symbol="circle", color="#1c83e1"),
#             hoverinfo="text",
#             hovertext=f"Duration: {phase['Duration']}<br>Topics: {phase['Topics']}"
#         ))
    
#     # Customize layout
#     fig.update_layout(
#         showlegend=False,
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)",
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         margin=dict(l=20, r=20, t=60, b=20),
#         height=200
#     )
    
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Display phase details
#     for phase, details in roadmap_data["timeline"].items():
#         with st.expander(f"📚 {details['name']} ({details['duration']})"):
#             st.markdown("#### Topics")
#             for topic in details["topics"]:
#                 st.markdown(f"- {topic}")
            
#             st.markdown("#### Projects")
#             for project in details["projects"]:
#                 st.markdown(f"- {project}")

# def render_profile_page():
#     st.markdown("## 👤 Your Profile")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         # Experience Level
#         st.markdown("### Experience Level")
#         experience = st.select_slider(
#             "Select your experience level",
#             options=["Beginner", "Intermediate", "Advanced"],
#             value=st.session_state.user_profile.experience_level.capitalize()
#         )
        
#         # Learning Style
#         st.markdown("### Learning Style")
#         learning_style = st.radio(
#             "Choose your preferred learning style",
#             ["Structured", "Project-based", "Theoretical"],
#             index=["structured", "project-based", "theoretical"].index(
#                 st.session_state.user_profile.learning_style
#             )
#         )
    
#     with col2:
#         # Interests
#         st.markdown("### Areas of Interest")
#         interests = st.multiselect(
#             "Select your areas of interest",
#             ["Machine Learning", "Deep Learning", "Data Analysis", 
#              "Computer Vision", "NLP", "Data Engineering", "MLOps"],
#             default=st.session_state.user_profile.interests
#         )
        
#         # Career Goals
#         st.markdown("### Career Goals")
#         career_goals = st.multiselect(
#             "Select your career goals",
#             ["Data Scientist", "ML Engineer", "Data Engineer", 
#              "Research Scientist", "AI Engineer", "MLOps Engineer"],
#             default=st.session_state.user_profile.career_goals
#         )
    
#     # Save button
#     if st.button("💾 Save Profile", key="save_profile"):
#         st.session_state.user_profile.experience_level = experience.lower()
#         st.session_state.user_profile.learning_style = learning_style.lower()
#         st.session_state.user_profile.interests = interests
#         st.session_state.user_profile.career_goals = career_goals
#         st.success("Profile updated successfully!")

# def render_chat_page():
#     st.markdown("## 💬 AI Academy Assistant")
    
#     # Chat interface
#     chat_container = st.container()
    
#     with chat_container:
#         for message in st.session_state.chat_history:
#             render_chat_message(
#                 message["content"],
#                 is_user=message["role"] == "user"
#             )
    
#     # Chat input
#     user_input = st.text_input(
#         "Ask me anything about data science, AI, or your learning path!",
#         key="user_input"
#     )
    
#     if user_input:
#         # Add user message
#         st.session_state.chat_history.append({
#             "role": "user",
#             "content": user_input
#         })
        
#         # Generate response
#         response = st.session_state.llm_handler.generate_response(
#             user_input,
#             st.session_state.user_profile,
#             st.session_state.chat_history
#         )
        
#         # Add assistant response
#         st.session_state.chat_history.append({
#             "role": "assistant",
#             "content": response["response"]
#         })
        
#         # Display roadmap if available
#         if response["roadmap"]:
#             render_roadmap(response["roadmap"])
        
#         # Clear input
#         st.experimental_rerun()

# def render_progress_page():
#     st.markdown("## 📈 Your Learning Progress")
    
#     # Sample progress data (replace with actual tracking logic)
#     progress_data = {
#         "courses_completed": 5,
#         "projects_completed": 3,
#         "skills_acquired": 12,
#         "study_hours": 45
#     }
    
#     # Metrics
#     col1, col2, col3, col4 = st.columns(4)
    
#     with col1:
#         st.markdown("""
#             <div class="metric-container">
#                 <div class="metric-value">5</div>
#                 <div class="metric-label">Courses Completed</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col2:
#         st.markdown("""
#             <div class="metric-container">
#                 <div class="metric-value">3</div>
#                 <div class="metric-label">Projects Completed</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col3:
#         st.markdown("""
#             <div class="metric-container">
#                 <div class="metric-value">12</div>
#                 <div class="metric-label">Skills Acquired</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     with col4:
#         st.markdown("""
#             <div class="metric-container">
#                 <div class="metric-value">45</div>
#                 <div class="metric-label">Study Hours</div>
#             </div>
#         """, unsafe_allow_html=True)
    
#     # Progress charts
#     st.markdown("### Skill Progress")
    
#     # Sample skill progress data
#     skills_data = {
#         "Python": 80,
#         "Machine Learning": 60,
#         "Data Analysis": 75,
#         "Deep Learning": 45,
#         "Statistics": 70
#     }
    
#     for skill, progress in skills_data.items():
#         col1, col2 = st.columns([1, 4])
#         with col1:
#             st.write(skill)
#         with col2:
#             st.progress(progress / 100)

# def main():
#     # Initialize
#     init_session_state()
#     load_css()
    
#     # Sidebar navigation
#     with st.sidebar:
#         st.markdown("# 🤖 AI Academy")
#         selected_page = option_menu(
#             "",
#             ["Chat", "Profile", "Progress"],
#             icons=["chat", "person", "graph-up"],
#             default_index=["Chat", "Profile", "Progress"].index(st.session_state.current_page)
#         )
#         st.session_state.current_page = selected_page
    
#     # Render selected page
#     if selected_page == "Chat":
#         render_chat_page()
#     elif selected_page == "Profile":
#         render_profile_page()
#     else:
#         render_progress_page()

# if __name__ == "__main__":
#     main()