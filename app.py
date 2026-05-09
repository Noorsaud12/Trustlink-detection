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
    else:
        followers = random.randint(50, 5000)
        following = random.randint(10, 1000)
        posts = random.randint(0, 10)
        age = random.randint(1, 2000)
        platform = random.choice(["Instagram", "Twitter", "Facebook"])
        images = random.randint(20, 200)
        st.write("⚠️ This account seems new, estimated values shown:")

    st.write(f"**Username:** {username}")
    st.write(f"**Platform:** {platform}")
    st.write(f"**Followers:** {followers}")
    st.write(f"**Following:** {following}")
    st.write(f"**Posts:** {posts}")
    st.write(f"**Account Age (days):** {age}")
    st.write(f"**Images Uploaded:** {images}")

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

    if reasons_fake:
        st.error("❌ This account is likely FAKE")
        st.write("Reasons:")
        for r in reasons_fake:
            st.write(r)
        st.write("📌 Summary: الحساب جديد جدًا مع نشاط قليل → Fake")
    else:
        st.success("✅ This account looks REAL")
        st.write("Reasons:")
        for r in reasons_real:
            st.write(r)
        st.write("📌 Summary: الحساب قديم ومتوازن → Real")
