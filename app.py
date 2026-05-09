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
        images = random.randint(5, 50)
        comments_enabled = random.choice([True, False])  # simulate comments status
    else:
        followers = random.randint(50, 5000)
        following = random.randint(10, 1000)
        posts = random.randint(0, 10)
        age = random.randint(1, 2000)
        platform = random.choice(["Instagram", "Twitter", "Facebook"])
        images = random.randint(20, 200)
        comments_enabled = random.choice([True, False])
        

    st.write(f"**Username:** {username}")
    st.write(f"**Platform:** {platform}")
    st.write(f"**Followers:** {followers}")
    st.write(f"**Following:** {following}")
    st.write(f"**Posts:** {posts}")
    st.write(f"**Account Age (days):** {age}")
    st.write(f"**Images Uploaded:** {images}")
    st.write(f"**Comments Status:** {'🔓 Open' if comments_enabled else '🔒 Closed'}")

    # Fake/Real logic with explanation
    reasons_fake = []
    reasons_real = []

    if age < 30:
        reasons_fake.append("🔴 Account age is very short (<30 days)")
    else:
        reasons_real.append("🟢 Account age is reasonable")

    if posts <= 2:
        reasons_fake.append("🔴 Very few posts (0–2)")
    else:
        reasons_real.append("🟢 Posts are consistent")

    if images > 50 and posts <= 2:
        reasons_fake.append("🔴 Too many images compared to posts")

    if followers < 100:
        reasons_fake.append("🔴 Very low followers count")
    else:
        reasons_real.append("🟢 Followers count looks fine")

    if followers/following < 0.5:
        reasons_fake.append("🔴 Followers/Following ratio is suspicious")
    else:
        reasons_real.append("🟢 Followers/Following ratio is balanced")

    if not comments_enabled:
        reasons_fake.append("🔴 Comments are disabled (no interaction)")
    else:
        reasons_real.append("🟢 Comments are open (normal interaction)")

    if reasons_fake:
        st.error("❌ This account is likely FAKE")
        st.write("Reasons:")
        for r in reasons_fake:
            st.write(r)
        st.write("📌 Summary: New account or suspicious activity + comments closed → Fake")
    else:
        st.success("✅ This account looks REAL")
        st.write("Reasons:")
        for r in reasons_real:
            st.write(r)
        st.write("📌 Summary: Old and balanced account + comments open → Real")

    # Bar chart visualization
    chart_data = pd.DataFrame({
        "Metric": ["Followers", "Following", "Posts", "Account Age (days)", "Images"],
        "Count": [followers, following, posts, age, images]
    })
    st.bar_chart(chart_data.set_index("Metric"))
