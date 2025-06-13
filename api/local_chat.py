#!/usr/bin/env python3
"""
Command-line interface for chatting with local repositories using the DeepWiki-Open RAG system.
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Generator, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary modules
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.schema import Document
    import faiss
except ImportError:
    logger.error("Required packages not found. Please install them using: pip install langchain faiss-cpu")
    sys.exit(1)

# Try to import embedding and model providers
PROVIDERS = {}

try:
    import google.generativeai as genai
    PROVIDERS["google"] = True
except ImportError:
    PROVIDERS["google"] = False
    logger.warning("Google Generative AI package not found. Google provider will not be available.")

try:
    import openai
    PROVIDERS["openai"] = True
except ImportError:
    PROVIDERS["openai"] = False
    logger.warning("OpenAI package not found. OpenAI provider will not be available.")

try:
    import requests
    PROVIDERS["openrouter"] = True
except ImportError:
    PROVIDERS["openrouter"] = False
    logger.warning("Requests package not found. OpenRouter provider will not be available.")

try:
    import boto3
    PROVIDERS["bedrock"] = True
except ImportError:
    PROVIDERS["bedrock"] = False
    logger.warning("Boto3 package not found. Bedrock provider will not be available.")

# Ollama doesn't require special packages, just the requests module
PROVIDERS["ollama"] = PROVIDERS.get("openrouter", False)


class LocalChat:
    """
    Class for chatting with local repositories using RAG.
    """
    
    def __init__(
        self,
        repo_path: str,
        exclude_dirs: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None,
        language: str = "en",
        provider: str = "google",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4000
    ):
        """
        Initialize the LocalChat instance.
        
        Args:
            repo_path: Path to the local repository
            exclude_dirs: List of directories to exclude
            exclude_files: List of files to exclude
            language: Language for responses
            provider: Model provider (google, openai, openrouter, ollama, bedrock)
            model: Model name (provider-specific)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for generation
        """
        self.repo_path = os.path.abspath(repo_path)
        self.exclude_dirs = exclude_dirs or ["node_modules", ".git", "__pycache__", "venv"]
        self.exclude_files = exclude_files or ["*.min.js", "*.map", "*.pyc"]
        self.language = language
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set default models based on provider if not specified
        if model is None:
            default_models = {
                "google": "gemini-2.0-flash",
                "openai": "gpt-4o",
                "openrouter": "openai/gpt-4o",
                "ollama": "llama3",
                "bedrock": "anthropic.claude-3-sonnet-20240229-v1:0"
            }
            self.model = default_models.get(provider, "")
        else:
            self.model = model
        
        # Initialize provider-specific clients
        self._init_provider()
        
        # Initialize RAG components
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Conversation history
        self.history = []
    
    def _init_provider(self):
        """Initialize the selected model provider."""
        if self.provider == "google":
            if not PROVIDERS["google"]:
                raise ImportError("Google Generative AI package not installed. Please install it with: pip install google-generativeai")
            
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")
            
            genai.configure(api_key=api_key)
            
            # Initialize embedding model
            self.embedding_model = "models/embedding-001"
            
            # Initialize generation model
            generation_config = {
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens,
                "top_p": 0.9,
            }
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            self.model_client = genai.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        
        elif self.provider == "openai":
            if not PROVIDERS["openai"]:
                raise ImportError("OpenAI package not installed. Please install it with: pip install openai")
            
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            
            self.client = openai.OpenAI(api_key=api_key)
            
            # Initialize embedding model
            self.embedding_model = "text-embedding-3-small"
        
        elif self.provider == "openrouter":
            if not PROVIDERS["openrouter"]:
                raise ImportError("Requests package not installed. Please install it with: pip install requests")
            
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OPENROUTER_API_KEY environment variable not set")
            
            self.api_key = api_key
            self.api_base = "https://openrouter.ai/api/v1"
            
            # Use OpenAI for embeddings
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable also needed for embeddings with OpenRouter")
            
            self.embedding_client = openai.OpenAI(api_key=openai_api_key)
            self.embedding_model = "text-embedding-3-small"
        
        elif self.provider == "ollama":
            if not PROVIDERS["ollama"]:
                raise ImportError("Requests package not installed. Please install it with: pip install requests")
            
            self.api_base = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            
            # Use OpenAI for embeddings if available, otherwise use a simple embedding function
            if PROVIDERS["openai"] and os.environ.get("OPENAI_API_KEY"):
                self.embedding_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                self.embedding_model = "text-embedding-3-small"
                self.use_openai_embeddings = True
            else:
                logger.warning("OpenAI API key not found. Using simple embedding function.")
                self.use_openai_embeddings = False
                # We'll implement a simple embedding function later
        
        elif self.provider == "bedrock":
            if not PROVIDERS["bedrock"]:
                raise ImportError("Boto3 package not installed. Please install it with: pip install boto3")
            
            region = os.environ.get("BEDROCK_REGION", "us-west-2")
            self.client = boto3.client("bedrock-runtime", region_name=region)
            
            # Use OpenAI for embeddings if available
            if PROVIDERS["openai"] and os.environ.get("OPENAI_API_KEY"):
                self.embedding_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                self.embedding_model = "text-embedding-3-small"
                self.use_openai_embeddings = True
            else:
                logger.warning("OpenAI API key not found. Using simple embedding function.")
                self.use_openai_embeddings = False
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.provider == "google":
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result["embedding"])
            return embeddings
        
        elif self.provider in ["openai", "openrouter"] or (
            self.provider in ["ollama", "bedrock"] and getattr(self, "use_openai_embeddings", False)
        ):
            client = getattr(self, "embedding_client", None) or self.client
            response = client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]
        
        else:
            # Simple embedding function for providers without embedding support
            # This is a very basic approach and won't work well in practice
            # In a real implementation, you would use a local embedding model
            logger.warning("Using simple embedding function. This is not recommended for production use.")
            
            def simple_embed(text):
                import hashlib
                import numpy as np
                
                # Create a hash of the text
                hash_obj = hashlib.md5(text.encode())
                hash_bytes = hash_obj.digest()
                
                # Convert hash to a list of floats
                hash_ints = [b for b in hash_bytes]
                
                # Create a 1536-dimensional vector (same as OpenAI's)
                vec = np.zeros(1536)
                for i, val in enumerate(hash_ints):
                    idx = i % 1536
                    vec[idx] += val / 255.0
                
                # Normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                
                return vec.tolist()
            
            return [simple_embed(text) for text in texts]
    
    def process_repository(self):
        """Process the repository and create embeddings."""
        logger.info(f"Processing repository: {self.repo_path}")
        
        # Check if repository exists
        if not os.path.isdir(self.repo_path):
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        
        # Collect files
        documents = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs and not any(
                os.path.join(root, d).startswith(os.path.join(self.repo_path, ex_dir))
                for ex_dir in self.exclude_dirs
            )]
            
            for file in files:
                # Skip excluded files
                if any(file.endswith(ex_file.replace("*", "")) for ex_file in self.exclude_files if "*" in ex_file) or \
                   any(file == ex_file for ex_file in self.exclude_files if "*" not in ex_file):
                    continue
                
                file_path = os.path.join(root, file)
                
                # Skip binary files and very large files
                if self._is_binary_file(file_path) or os.path.getsize(file_path) > 1000000:  # 1MB
                    continue
                
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    rel_path = os.path.relpath(file_path, self.repo_path)
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": rel_path}
                    ))
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {str(e)}")
        
        logger.info(f"Found {len(documents)} documents")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Create embeddings and vectorstore
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        embeddings = self._get_embeddings(texts)
        
        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(faiss.np.array(embeddings, dtype=faiss.np.float32))
        
        # Create vectorstore
        self.vectorstore = FAISS(
            embedding_function=self._get_embeddings,
            index=index,
            docstore={"texts": texts, "metadatas": metadatas},
            index_to_docstore_id={i: i for i in range(len(texts))}
        )
        
        logger.info("Repository processing complete")
    
    def _is_binary_file(self, file_path: str) -> bool:
        """
        Check if a file is binary.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is binary, False otherwise
        """
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
                return b"\0" in chunk
        except Exception:
            return True
    
    def _retrieve_context(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: The query to retrieve context for
            k: Number of documents to retrieve
            
        Returns:
            Retrieved context as a string
        """
        if self.vectorstore is None:
            raise ValueError("Repository not processed. Call process_repository() first.")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        
        context = "\n\n".join([
            f"File: {doc.metadata['source']}\n{doc.page_content}"
            for doc in docs
        ])
        
        return context
    
    def ask(self, query: str) -> str:
        """
        Ask a question about the repository.
        
        Args:
            query: The question to ask
            
        Returns:
            The response from the model
        """
        # Retrieve context
        context = self._retrieve_context(query)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate response
        response = self._generate_response(prompt)
        
        # Update history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def ask_stream(self, query: str) -> Generator[str, None, None]:
        """
        Ask a question and stream the response.
        
        Args:
            query: The question to ask
            
        Yields:
            Chunks of the response
        """
        # Retrieve context
        context = self._retrieve_context(query)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate streaming response
        response_text = ""
        for chunk in self._generate_response_stream(prompt):
            response_text += chunk
            yield chunk
        
        # Update history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response_text})
    
    def _create_prompt(self, query: str, context: str) -> Dict[str, Any]:
        """
        Create a prompt for the model.
        
        Args:
            query: The question to ask
            context: The retrieved context
            
        Returns:
            Prompt in the format expected by the model
        """
        system_message = f"""You are an AI assistant that helps developers understand code repositories. 
You have access to code snippets from a repository, and your task is to answer questions about the code.
Always base your answers on the provided context. If you don't know the answer, say so.
Respond in {self.language} language.

Here is the context from the repository:

{context}"""
        
        # Format based on provider
        if self.provider == "google":
            return [
                {"role": "system", "parts": [{"text": system_message}]},
                *[{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in self.history],
                {"role": "user", "parts": [{"text": query}]}
            ]
        
        elif self.provider in ["openai", "openrouter"]:
            return [
                {"role": "system", "content": system_message},
                *self.history,
                {"role": "user", "content": query}
            ]
        
        elif self.provider == "ollama":
            history_text = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in self.history
            ])
            
            return {
                "system": system_message,
                "prompt": f"{history_text}\nUser: {query}\nAssistant:",
                "model": self.model,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
        
        elif self.provider == "bedrock":
            if "claude" in self.model.lower():
                # Claude format
                history_text = "\n\n".join([
                    f"Human: {msg['content']}" if msg['role'] == "user" else f"Assistant: {msg['content']}"
                    for msg in self.history
                ])
                
                prompt = f"{system_message}\n\n{history_text}\n\nHuman: {query}\n\nAssistant:"
                
                return {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "messages": [
                        {"role": "system", "content": system_message},
                        *self.history,
                        {"role": "user", "content": query}
                    ]
                }
            else:
                # Generic format for other Bedrock models
                history_text = "\n\n".join([
                    f"User: {msg['content']}" if msg['role'] == "user" else f"Assistant: {msg['content']}"
                    for msg in self.history
                ])
                
                prompt = f"{system_message}\n\n{history_text}\n\nUser: {query}\n\nAssistant:"
                
                return {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": self.max_tokens,
                        "temperature": self.temperature,
                        "topP": 0.9
                    }
                }
    
    def _generate_response(self, prompt: Any) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: The prompt for the model
            
        Returns:
            The generated response
        """
        if self.provider == "google":
            response = self.model_client.generate_content(prompt)
            return response.text
        
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        
        elif self.provider == "openrouter":
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenRouter API error: {response.text}")
            
            return response.json()["choices"][0]["message"]["content"]
        
        elif self.provider == "ollama":
            response = requests.post(
                f"{self.api_base}/api/generate",
                json=prompt
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.text}")
            
            return response.json()["response"]
        
        elif self.provider == "bedrock":
            import json
            
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps(prompt)
            )
            
            response_body = json.loads(response["body"].read())
            
            if "anthropic" in self.model.lower():
                return response_body["content"][0]["text"]
            else:
                return response_body.get("outputText", "")
    
    def _generate_response_stream(self, prompt: Any) -> Generator[str, None, None]:
        """
        Generate a streaming response from the model.
        
        Args:
            prompt: The prompt for the model
            
        Yields:
            Chunks of the generated response
        """
        if self.provider == "google":
            response = self.model_client.generate_content(prompt, stream=True)
            for chunk in response:
                if hasattr(chunk, "text"):
                    yield chunk.text
        
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        elif self.provider == "openrouter":
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "messages": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": True
                },
                stream=True
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenRouter API error: {response.text}")
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: ") and not line.startswith("data: [DONE]"):
                        data = json.loads(line[6:])
                        if data["choices"] and data["choices"][0]["delta"].get("content"):
                            yield data["choices"][0]["delta"]["content"]
        
        else:
            # For providers that don't support streaming, yield the entire response at once
            yield self._generate_response(prompt)
    
    def clear_history(self):
        """Clear the conversation history."""
        self.history = []


def main():
    """Main function for the command-line interface."""
    parser = argparse.ArgumentParser(description="Chat with a local repository using DeepWiki-Open RAG")
    parser.add_argument("--repo_path", required=True, help="Path to the local repository")
    parser.add_argument("--exclude_dirs", help="Directories to exclude (comma-separated)")
    parser.add_argument("--exclude_files", help="Files to exclude (comma-separated)")
    parser.add_argument("--language", default="en", help="Language for responses")
    parser.add_argument("--provider", default="google", 
                        choices=["google", "openai", "openrouter", "ollama", "bedrock"],
                        help="Model provider")
    parser.add_argument("--model", help="Model name (provider-specific)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Maximum tokens for generation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Convert comma-separated strings to lists
    exclude_dirs = args.exclude_dirs.split(",") if args.exclude_dirs else None
    exclude_files = args.exclude_files.split(",") if args.exclude_files else None
    
    try:
        # Initialize chat
        chat = LocalChat(
            repo_path=args.repo_path,
            exclude_dirs=exclude_dirs,
            exclude_files=exclude_files,
            language=args.language,
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # Process repository
        print(f"Processing repository at {args.repo_path}...")
        chat.process_repository()
        print("Repository processed successfully!")
        
        # Interactive chat loop
        print("\n=== DeepWiki-Open Chat ===")
        print(f"Repository: {args.repo_path}")
        print(f"Model: {args.provider}/{chat.model}")
        print("Type 'exit', 'quit', or 'q' to end the conversation.")
        print("Type 'clear' to clear the conversation history.")
        print("=" * 30)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == "clear":
                    chat.clear_history()
                    print("Conversation history cleared.")
                    continue
                
                if not user_input:
                    continue
                
                print("\nAI: ", end="", flush=True)
                
                # Get response with streaming if supported
                if hasattr(chat, "ask_stream") and args.provider in ["google", "openai", "openrouter"]:
                    # Stream the response
                    for chunk in chat.ask_stream(user_input):
                        print(chunk, end="", flush=True)
                    print()  # Add a newline at the end
                else:
                    # Get the full response at once
                    response = chat.ask(user_input)
                    print(response)
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
