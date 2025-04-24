#app.py
import streamlit as st
import pandas as pd
from utils import mask_pii
from models import classify_email

# Configure page
st.set_page_config(
    page_title="Email Classification System",
    page_icon="📧",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .header {
        font-size: 24px !important;
        color: #1f77b4;
    }
    .highlight {
        background-color: #e6f3ff;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .entity-table {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("📧 Email Classification & PII Masking System")
    st.markdown("---")
    
    # Input section
    email_input = st.text_area(
        "Paste your email content here:",
        height=200,
        placeholder="Enter email text..."
    )
    
    if st.button("Process Email", help="Click to process the email"):
        if email_input.strip():
            with st.spinner("Processing email..."):
                # Mask PII
                masked_email, masked_entities = mask_pii(email_input)
                
                # Classify email
                category = classify_email(masked_email)
                
                # Display results
                st.markdown("### Results")
                
                # Category display
                st.markdown(f"**Email Category:** <span style='color:green'>{category}</span>", 
                           unsafe_allow_html=True)
                
                # Masked email display
                st.markdown("**Masked Email Content:**")
                st.markdown(f'<div class="highlight">{masked_email}</div>', 
                           unsafe_allow_html=True)
                
                # Masked entities table
                if masked_entities:
                    st.markdown("**Detected PII Entities:**")
                    df = pd.DataFrame([{
                        "Entity Type": ent['type'],
                        "Original Value": ent['value'],
                        "Position": f"{ent['start']}-{ent['end']}"
                    } for ent in masked_entities])
                    
                    st.dataframe(
                        df,
                        column_config={
                            "Entity Type": "Entity Type",
                            "Original Value": "Original Value",
                            "Position": "Text Position"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                else:
                    st.info("No PII entities detected in this email")
        else:
            st.warning("Please enter some email content to process")

if __name__ == "__main__":
    main()