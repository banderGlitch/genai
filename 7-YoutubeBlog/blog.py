import streamlit as st
from youtubeblog import (
    YoutubeBlogState,
    extract_transcript,
    summarize_transcript,
    generate_blog_post,
    human_review,
    should_continue,
)
# Using streamlit to create a web app

st.title("Youtube Blog Generator")

# Input: Ask user for YouTube video link.
video_url = st.text_input("Enter YouTube Video URL:")

if video_url:
    st.info("Processing... please wait.")
    state: YoutubeBlogState = {"video_url": video_url}
    # Run the workflow steps (excluding human_review, which we replace with a Streamlit UI)
    
    try:
        state = extract_transcript(state)
        state = summarize_transcript(state)
        state = generate_blog_post(state)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error(traceback.format_exc())
    

    
    





