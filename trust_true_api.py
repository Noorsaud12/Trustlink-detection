import streamlit as st
import pandas as pd
import random

# Sample dataset
data = {
    "username": ["noor123", "trustlink22", "ahmed_dev"],
    "followers": [150, 2000, 500],
    "following": [100, 300, 50],
    "posts": [20, 150, 40],
    "account_age_days": [365, 120, 600],
    "platform": ["Instagram", "Twitter", "Facebook"]
}

df = pd.DataFrame(data)

st.title("TrustLink Fake Detection")

username = st.text_input("Enter username:")

if username:
    user = df[df["username"] == username]

    if not user.empty:
        followers = user['followers'].values[0]
        following = user['following'].values[0]
        posts = user['posts'].values[0]
        age = user['account_age_days'].values[0]
        platform = user['platform'].values[0]
        images = random.randint(5, 50)  # simulate image count
    else:
        # Random values for unknown usernames
        followers = random.randint(50, 5000)
        following = random.randint(10, 1000)
        posts = random.randint(0, 10)
        age = random.randint(1, 2000)
        platform = random.choice(["Instagram", "Twitter", "Facebook"])
        images = random.randint(20, 200)  # suspiciously high images
        

    # Show data
    st.write(f"**Username:** {username}")
    st.write(f"**Platform:** {platform}")
    st.write(f"**Followers:** {followers}")
    st.write(f"**Following:** {following}")
    st.write(f"**Posts:** {posts}")
    st.write(f"**Account Age (days):** {age}")
    st.write(f"**Images Uploaded:** {images}")

    # Fake/Real logic
    if age < 30 and posts <= 2:
        st.error("❌ This account is likely FAKE (new account, almost no posts)")
    elif posts <= 2 and images > 50:
        st.error("❌ This account is likely FAKE (too many images, no real posts)")
    elif followers < 100 and posts < 5:
        st.warning("⚠️ Suspicious account (possible fake)")
    else:
        st.success("✅ This account looks REAL")

    # Bar chart visualization
    chart_data = pd.DataFrame({
        "Metric": ["Followers", "Following", "Posts", "Account Age (days)", "Images"],
        "Count": [followers, following, posts, age, images]
    })
    st.bar_chart(chart_data.set_index("Metric"))
