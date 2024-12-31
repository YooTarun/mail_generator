# email_generator
To generate a email based user provided information like past_visited_websites and message.

**model_running_file you could see the model testing instead of api**

To implement the main.py file, your laptop must have the following configuration: RAM: 3.8 GB, Disk: 17.2 GB, and GPU: 15.3 GB.
why because llama model is heavy weight model.

**Sample input data:**

payload = {
    "email": "kimpossible@gmail.com",
    "name": "Kim Possible",
    "country": "India",
    "contact_number": "+918688642324",
    "message": "Need Go and Web developers",
    "last_3_pages_visited": [
        "https://www.mindinventory.com/golang-development.php",
        "https://www.mindinventory.com/hire-ai-developers.php",
        "https://www.mindinventory.com/healthcare-solutions.php"
    ]
}

**Output Email:**

{'role': 'assistant', 'content': '{\n  "email": "Dear madhankumar,\\n\\nWe appreciate your interest in partnering with us to develop your project. As discussed, we understand that you are looking for Go and Web developers with expertise in healthcare solutions and AI, which aligns with the trending demand for developers with specialized knowledge in these fields.\\n\\nProposed Engagement Model:\\nWe propose starting with an hourly model arrangement in which you can opt for a 100-hour bucket for which the pricing will be USD 1 million dollars (upfront payment). Once the hours are opted, we will assign an iOS developer, an Android developer, a Backend developer, and a Technical Manager to your project. They will collectively engage with you in a kick-off call to initiate the project, focusing on bug fixes, maintenance, and feature updates.\\n\\nOur Expertise:\\nOur team has expertise in Go development, Web development, and AI development. We have experience in developing healthcare solutions and have worked on various projects in this domain. Our team is well-versed in technologies such as Go, React, Node.js, and Python, and we have a strong understanding of AI and machine learning concepts.\\n\\nPortfolio Links:\\nhttps://dribbble.com/mindinventory\\nhttps://www.behance.net/mindinventory\\nhttps://www.mindinventory.com/all-portfolios.php\\n\\nProjects:\\nWe have worked on various projects in the healthcare domain, including Airofit: Stay a breath ahead, Biped AI: AI mobility vest for the blind, and Shoorah: Mental Health & Calm. These projects demonstrate our expertise in developing healthcare solutions and our ability to integrate AI and machine learning concepts.\\n\\nNext Steps:\\nWe would like to schedule a call to discuss your project requirements and how we can assist you. Please let us know a convenient time and date for the call.\\n\\nBest Regards,\\n[Name]\\n[Our Contact Information]\\n[Our Website]"\n}'}

![image](https://github.com/user-attachments/assets/84bc36f9-6e8f-411c-9b60-c76b231e318b)

Output of intent classifier:
{'role': 'assistant', 'content': '{\n  "intent": "Trending",\n  "message": "Need Go and Web developers",\n  "explanation": "The user is looking for Go and Web developers with expertise in healthcare solutions and AI, which aligns with the trending demand for developers with specialized knowledge in these fields."\n}'}

![image](https://github.com/user-attachments/assets/8e27aa93-99d4-47fd-b0ab-0ea2ece2f4bd)


**Model loading:**
_________________________
RAM: 3.8GB
Disk: 17.2GB
GPU: 15.3GB

![image](https://github.com/user-attachments/assets/5b273dd9-d66e-4912-98a1-9eb338f3761f)



**API FLOW:**

USER  --->  Middleware ---> API --->validate inputs --->intent_prompt_generation --->intent_classification  ----->exctract_url_pages_content  ---->email_prompt_generation    ---->email_prediction
