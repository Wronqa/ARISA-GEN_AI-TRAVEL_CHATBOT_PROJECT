import streamlit as st
import time
import nest_asyncio
from langchain.memory import ConversationBufferMemory


import config
import llm_services
import search
import validation
import chatbot
import find_travel_tool 

nest_asyncio.apply()


st.set_page_config(
    page_title="TravelBotFlex Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


langchain_llm = llm_services.setup_langchain_llm()
st_model = llm_services.load_st_model()
faq_embeddings, faq_segments, faq_categories = llm_services.load_faq_data()


if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    print("Initialized new LangChain conversation memory.")


if "messages" not in st.session_state:
    st.session_state.messages = []

    welcome_message = """Hi! I'm TravelBotFlex, your travel assistant.

I can help you find information about our offers and answer travel-related questions based on our internal knowledge.
How can I help you today? 😊"""
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})


if "show_offer_dialog" not in st.session_state:
    st.session_state.show_offer_dialog = False
if "offer_search_result" not in st.session_state:
    st.session_state.offer_search_result = ""



with st.sidebar:
    st.title("TravelBotFlex ✈️")
    st.markdown("---")
    st.markdown("**Your intelligent travel assistant!**")
    st.markdown("Ask me your travel questions in the main chat.")
    st.markdown("---")
    st.markdown(f"Using model: `{config.GEMINI_MODEL_NAME}`") 
    st.markdown("Remember, I'm still under development!")
    st.markdown("---")
    if st.button("🗑️ Clear Chat History"):

        st.session_state.messages = []

        if "conversation_memory" in st.session_state:
            st.session_state.conversation_memory.clear()
            print("LangChain conversation memory cleared.")
   
        welcome_message = """Hi! I'm TravelBotFlex, your travel assistant. How can I help you today? 😊"""
        st.session_state.messages.append({"role": "assistant", "content": welcome_message})
      
        st.session_state.show_offer_dialog = False
        st.session_state.offer_search_result = ""
        st.rerun()


    st.markdown("---")
    st.subheader("🔎 Find Offer by Code")
    offer_code_input = st.text_input("Enter Offer Code (e.g., CUB-HAV26):", key="offer_code_input")

  
    if st.button("Search Offer", key="search_offer_button"):
        if offer_code_input:
            with st.spinner("Searching for offer..."):

                search_result = find_travel_tool.offer_search_tool.run(offer_code_input)
 
                st.session_state.offer_search_result = search_result
                st.session_state.show_offer_dialog = True
 
                st.rerun()
        else:
            st.warning("Please enter an offer code to search.")
    st.markdown("---")


if st.session_state.get("show_offer_dialog", False):
    result_content = st.session_state.get("offer_search_result", "Error: Search result not found.")


    @st.dialog("Offer Search Result")
    def show_result_dialog():
        st.markdown(result_content)
        if st.button("Close", key="dialog_close_button"):
            st.session_state.show_offer_dialog = False
            st.rerun() 


    show_result_dialog()



st.header("Chat with TravelBotFlex 🤖")


for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask TravelBotFlex about your trip..."):
   
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)


    final_response = None
    blocked = False
    blocked_message = "An error occurred." 

 
    if not validation.validate_input_basic(prompt):
        blocked = True
        blocked_message = "Your message couldn't be processed due to potentially problematic content found by basic checks. Please rephrase."
    else:
        with st.spinner("Analyzing request safety..."):
            is_safe = validation.evaluate_prompt_safety(prompt, langchain_llm)
        if not is_safe:
            blocked = True
            blocked_message = "I cannot process this request as it might be inappropriate, harmful, or outside my capabilities. Please ask a standard travel-related question."

 
    if not blocked:
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🧠 Finding information and formulating response...")

           
            start_time = time.time()
            faq_results = search.find_similar_faq(
                user_question=prompt,
                embeddings=faq_embeddings,
                segments=faq_segments,
                categories=faq_categories,
                model=st_model
          
            )
            search_time = time.time() - start_time
            print(f"FAQ Search Time: {search_time:.2f}s, Found: {len(faq_results)}")

          
            start_time = time.time()
            raw_response = chatbot.generate_langchain_response(
                query=prompt,
                faq_results=faq_results,
                llm=langchain_llm,
                memory=st.session_state.conversation_memory
            )
            generation_time = time.time() - start_time
            print(f"LangChain Main Generation Time: {generation_time:.2f}s")

        
            if validation.validate_output(raw_response):
                final_response = raw_response
              
                try:
                    st.session_state.conversation_memory.save_context({"input": prompt}, {"output": final_response})
                    print("Interaction saved to memory.")
                except Exception as e:
                    st.error(f"Error saving context to LangChain memory: {e}")
            else:
                blocked = True 
                blocked_message = "I apologize, I seem to have generated an unexpected response. Could you please try rephrasing your question?"
                st.warning("Blocked potentially inappropriate or invalid AI response.")
                final_response = blocked_message 

            
            message_placeholder.markdown(final_response)

    
    if blocked and final_response is None: 
        with st.chat_message("assistant", avatar="🤖"):
            st.warning(blocked_message) 
        final_response = blocked_message 

    
    if final_response:
        st.session_state.messages.append({"role": "assistant", "content": final_response})
    else:
        error_msg = "Sorry, something went wrong processing your request."
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant", avatar="🤖"):
             st.error(error_msg)