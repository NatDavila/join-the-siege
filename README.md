# Submission - Brief 

I trained a model using a simple Pipeline with a TF-IDF Vectorizer and Logistic Regression classifier. While GPT, BERT, or BART could be used, they are costly and computationally expensive for a straightforward classification task. The model classifies documents as invoices, bank statements, or licenses and has been improved to support additional file types like Word docs, .txt, and Excel files. This approach offers efficiency without unnecessary complexity.

The model was trained with a balanced dataset of 52 examples covering invoices, bank statements, and licenses. While larger datasets are typically preferred, I used a smaller set due to time constraints and still achieved 100% accuracy.

The model includes a safeguard to prevent accidental retraining, ensuring training only occurs when explicitly executed. Errors are logged for troubleshooting, maintaining consistency, and control over the retraining process.

The utils module handles text extraction from files, while the classifier module leverages the extractor to process the text and utilize the model for classification. This structure maintains a clear distinction between the text extraction and classification processes.

Key test cases were developed for these modules, though they are not exhaustive due to time constraints for the assignment.

To enhance production readiness, Docker was used to containerize the application, ensuring consistent dependencies, seamless deployment across environments, and improved scalability.

## To Run

First, build the docker image with
```
make docker-build
```
And then run the flask app with
```
make docker-run
```

You can then make requests against it at 
```
curl -X POST -F 'file=@path_to_pdf.pdf' http://127.0.0.1:5001/classify_file
```

# Heron Coding Challenge - File Classifier

## Overview

At Heron, we’re using AI to automate document processing workflows in financial services and beyond. Each day, we handle over 100,000 documents that need to be quickly identified and categorised before we can kick off the automations.

This repository provides a basic endpoint for classifying files by their filenames. However, the current classifier has limitations when it comes to handling poorly named files, processing larger volumes, and adapting to new industries effectively.

**Your task**: improve this classifier by adding features and optimisations to handle (1) poorly named files, (2) scaling to new industries, and (3) processing larger volumes of documents.

This is a real-world challenge that allows you to demonstrate your approach to building innovative and scalable AI solutions. We’re excited to see what you come up with! Feel free to take it in any direction you like, but we suggest:


### Part 1: Enhancing the Classifier

- What are the limitations in the current classifier that's stopping it from scaling?
- How might you extend the classifier with additional technologies, capabilities, or features?


### Part 2: Productionising the Classifier 

- How can you ensure the classifier is robust and reliable in a production environment?
- How can you deploy the classifier to make it accessible to other services and users?

We encourage you to be creative! Feel free to use any libraries, tools, services, models or frameworks of your choice

### Possible Ideas / Suggestions
- Train a classifier to categorize files based on the text content of a file
- Generate synthetic data to train the classifier on documents from different industries
- Detect file type and handle other file formats (e.g., Word, Excel)
- Set up a CI/CD pipeline for automatic testing and deployment
- Refactor the codebase to make it more maintainable and scalable

## Marking Criteria
- **Functionality**: Does the classifier work as expected?
- **Scalability**: Can the classifier scale to new industries and higher volumes?
- **Maintainability**: Is the codebase well-structured and easy to maintain?
- **Creativity**: Are there any innovative or creative solutions to the problem?
- **Testing**: Are there tests to validate the service's functionality?
- **Deployment**: Is the classifier ready for deployment in a production environment?


## Getting Started
1. Clone the repository:
    ```shell
    git clone <repository_url>
    cd heron_classifier
    ```

2. Install dependencies:
    ```shell
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Run the Flask app:
    ```shell
    python -m src.app
    ```

4. Test the classifier using a tool like curl:
    ```shell
    curl -X POST -F 'file=@path_to_pdf.pdf' http://127.0.0.1:5000/classify_file
    ```

5. Run tests:
   ```shell
    pytest
    ```

## Submission

Please aim to spend 3 hours on this challenge.

Once completed, submit your solution by sharing a link to your forked repository. Please also provide a brief write-up of your ideas, approach, and any instructions needed to run your solution. 
