#!/usr/bin/env python3
"""
Knowledge Graph Builder for Notion Hierarchical Data
Pulls data from Notion database and creates a knowledge graph structure
preserving parent-child relationships and KB article connections.

Dependencies for LangChain 1.0:
    pip install langchain-core           # For Document conversion
    pip install langchain-chroma         # For ChromaDB vector store  
    pip install langchain-openai         # For OpenAI embeddings
    
Environment Variables:
    NOTION_TOKEN    - Notion API token
    OPENAI_API_KEY  - OpenAI API key for embeddings
"""

import os
import json
import requests
from dotenv import load_dotenv
from typing import Dict, List, Optional, Set, Tuple
import networkx as nx
from datetime import datetime

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Create a simple Document class for type hints if langchain is not available
    class Document:
        def __init__(self, page_content: str, metadata: Dict, id: Optional[str] = None):
            self.page_content = page_content
            self.metadata = metadata
            self.id = id

try:
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Load environment variables
load_dotenv()

# Initialize Notion API headers
NOTION_TOKEN = os.getenv("NOTION_TOKEN")
if not NOTION_TOKEN:
    raise ValueError("NOTION_TOKEN not found in .env file")

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

# Target database ID
DATABASE_ID = "21b1db9ba441800cb53ce88a2a988680"

# ChromaDB Configuration (LangChain 1.0)
CHROMA_COLLECTION_NAME = "notion_knowledge_graph"
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model


class NotionKnowledgeGraphBuilder:
    """Build a knowledge graph from hierarchical Notion data"""
    
    def __init__(self, database_id: str):
        self.database_id = database_id
        self.graph = nx.DiGraph()  # Directed graph for parent-child relationships
        self.pages_cache = {}  # Cache for fetched pages
        self.kb_articles_cache = {}  # Cache for KB articles
        
    def fetch_database_pages(self) -> List[Dict]:
        """
        Fetch all pages from the Notion database
        
        Returns:
            List[Dict]: List of page objects
        """
        all_pages = []
        has_more = True
        start_cursor = None
        
        print(f"Fetching pages from database: {self.database_id}")
        
        try:
            while has_more:
                payload = {}
                if start_cursor:
                    payload["start_cursor"] = start_cursor
                
                url = f"https://api.notion.com/v1/databases/{self.database_id}/query"
                response = requests.post(url, headers=HEADERS, json=payload)
                response.raise_for_status()
                
                data = response.json()
                pages = data.get("results", [])
                all_pages.extend(pages)
                
                has_more = data.get("has_more", False)
                start_cursor = data.get("next_cursor")
                
                print(f"  Fetched {len(pages)} pages (total: {len(all_pages)})")
            
            print(f"✓ Successfully fetched {len(all_pages)} total pages\n")
            return all_pages
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching database pages: {e}")
            raise
    
    def extract_property_value(self, prop: Dict) -> any:
        """
        Extract the actual value from a Notion property
        
        Args:
            prop (Dict): Property object from Notion
            
        Returns:
            any: Extracted value
        """
        prop_type = prop.get("type")
        
        if prop_type == "title":
            return "".join([text["plain_text"] for text in prop.get("title", [])])
        
        elif prop_type == "rich_text":
            return "".join([text["plain_text"] for text in prop.get("rich_text", [])])
        
        elif prop_type == "number":
            return prop.get("number")
        
        elif prop_type == "select":
            select = prop.get("select")
            return select.get("name") if select else None
        
        elif prop_type == "multi_select":
            return [item["name"] for item in prop.get("multi_select", [])]
        
        elif prop_type == "url":
            return prop.get("url")
        
        elif prop_type == "relation":
            # Return list of relation IDs
            return [item["id"] for item in prop.get("relation", [])]
        
        elif prop_type == "rollup":
            rollup_data = prop.get("rollup", {})
            rollup_type = rollup_data.get("type")
            
            if rollup_type == "array":
                return [self._extract_rollup_item(item) for item in rollup_data.get("array", [])]
            elif rollup_type == "number":
                return rollup_data.get("number")
            else:
                return None
        
        elif prop_type == "formula":
            formula_data = prop.get("formula", {})
            formula_type = formula_data.get("type")
            
            if formula_type == "string":
                return formula_data.get("string")
            elif formula_type == "number":
                return formula_data.get("number")
            elif formula_type == "boolean":
                return formula_data.get("boolean")
            else:
                return None
        
        else:
            return prop.get(prop_type)
    
    def _extract_rollup_item(self, item: Dict) -> any:
        """Extract value from a rollup array item"""
        item_type = item.get("type")
        
        if item_type == "rich_text":
            return "".join([text["plain_text"] for text in item.get("rich_text", [])])
        elif item_type == "number":
            return item.get("number")
        elif item_type == "title":
            return "".join([text["plain_text"] for text in item.get("title", [])])
        else:
            return None
    
    def fetch_kb_article_metadata(self, page_id: str) -> Optional[Dict]:
        """
        Fetch metadata and content for a KB article page
        
        Args:
            page_id (str): Page ID of the KB article
            
        Returns:
            Optional[Dict]: KB article metadata with full content or None
        """
        if page_id in self.kb_articles_cache:
            return self.kb_articles_cache[page_id]
        
        try:
            url = f"https://api.notion.com/v1/pages/{page_id}"
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            
            page = response.json()
            
            # Extract basic metadata
            metadata = {
                "id": page["id"],
                "url": page.get("url"),
                "created_time": page.get("created_time"),
                "last_edited_time": page.get("last_edited_time"),
                "title": self._extract_page_title(page),
                "properties": {}
            }
            
            # Extract key properties
            properties = page.get("properties", {})
            for prop_name, prop_value in properties.items():
                metadata["properties"][prop_name] = self.extract_property_value(prop_value)
            
            # Fetch the actual page content (blocks)
            print(f"    Fetching content for KB article: {metadata['title']}")
            page_content = self.fetch_page_content(page_id)
            metadata["page_content"] = page_content
            
            self.kb_articles_cache[page_id] = metadata
            return metadata
            
        except requests.exceptions.RequestException as e:
            print(f"  Warning: Could not fetch KB article {page_id}: {e}")
            return None
    
    def _extract_page_title(self, page: Dict) -> str:
        """Extract title from a page object"""
        properties = page.get("properties", {})
        
        # Try common title property names
        for title_key in ["Name", "Title", "title", "name"]:
            if title_key in properties:
                prop = properties[title_key]
                if prop.get("type") == "title":
                    return "".join([text["plain_text"] for text in prop.get("title", [])])
        
        return "Untitled"
    
    def _extract_block_text(self, block: Dict) -> str:
        """
        Extract text content from a Notion block
        
        Args:
            block (Dict): A Notion block object
            
        Returns:
            str: Extracted text with basic formatting
        """
        block_type = block.get("type")
        
        # Handle different block types
        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", 
                          "bulleted_list_item", "numbered_list_item", "quote", 
                          "callout", "toggle"]:
            block_data = block.get(block_type, {})
            rich_text = block_data.get("rich_text", [])
            
            # Extract text and preserve inline links
            text_parts = []
            for text_obj in rich_text:
                plain_text = text_obj.get("plain_text", "")
                # Check if this text has a link
                if text_obj.get("href"):
                    text_parts.append(f"[{plain_text}]({text_obj.get('href')})")
                else:
                    text_parts.append(plain_text)
            
            text = "".join(text_parts)
            
            # Add formatting based on block type
            if block_type == "heading_1":
                return f"# {text}"
            elif block_type == "heading_2":
                return f"## {text}"
            elif block_type == "heading_3":
                return f"### {text}"
            elif block_type in ["bulleted_list_item", "numbered_list_item"]:
                return f"• {text}"
            elif block_type == "quote":
                return f"> {text}"
            else:
                return text
        
        elif block_type == "code":
            code_block = block.get("code", {})
            rich_text = code_block.get("rich_text", [])
            code = "".join([t.get("plain_text", "") for t in rich_text])
            language = code_block.get("language", "")
            return f"```{language}\n{code}\n```"
        
        elif block_type == "divider":
            return "---"
        
        elif block_type == "to_do":
            to_do = block.get("to_do", {})
            rich_text = to_do.get("rich_text", [])
            text = "".join([t.get("plain_text", "") for t in rich_text])
            checked = "✓" if to_do.get("checked") else "☐"
            return f"{checked} {text}"
        
        elif block_type == "image":
            image_data = block.get("image", {})
            # Images can be external or hosted by Notion
            if image_data.get("type") == "external":
                url = image_data.get("external", {}).get("url", "")
            elif image_data.get("type") == "file":
                url = image_data.get("file", {}).get("url", "")
            else:
                url = ""
            
            # Get caption if available
            caption_parts = image_data.get("caption", [])
            caption = "".join([t.get("plain_text", "") for t in caption_parts])
            
            if caption:
                return f"[IMAGE: {caption}]({url})"
            else:
                return f"[IMAGE]({url})"
        
        elif block_type == "video":
            video_data = block.get("video", {})
            # Videos can be external (YouTube, Vimeo) or uploaded files
            if video_data.get("type") == "external":
                url = video_data.get("external", {}).get("url", "")
            elif video_data.get("type") == "file":
                url = video_data.get("file", {}).get("url", "")
            else:
                url = ""
            
            # Get caption if available
            caption_parts = video_data.get("caption", [])
            caption = "".join([t.get("plain_text", "") for t in caption_parts])
            
            if caption:
                return f"[VIDEO: {caption}]({url})"
            else:
                return f"[VIDEO]({url})"
        
        elif block_type == "file":
            file_data = block.get("file", {})
            # Files can be external or uploaded to Notion
            if file_data.get("type") == "external":
                url = file_data.get("external", {}).get("url", "")
            elif file_data.get("type") == "file":
                url = file_data.get("file", {}).get("url", "")
            else:
                url = ""
            
            # Get caption if available
            caption_parts = file_data.get("caption", [])
            caption = "".join([t.get("plain_text", "") for t in caption_parts])
            
            if caption:
                return f"[FILE: {caption}]({url})"
            else:
                return f"[FILE]({url})"
        
        elif block_type == "embed":
            embed_data = block.get("embed", {})
            url = embed_data.get("url", "")
            
            # Get caption if available
            caption_parts = embed_data.get("caption", [])
            caption = "".join([t.get("plain_text", "") for t in caption_parts])
            
            if caption:
                return f"[EMBED: {caption}]({url})"
            else:
                return f"[EMBED]({url})"
        
        elif block_type == "pdf":
            pdf_data = block.get("pdf", {})
            # PDFs can be external or uploaded
            if pdf_data.get("type") == "external":
                url = pdf_data.get("external", {}).get("url", "")
            elif pdf_data.get("type") == "file":
                url = pdf_data.get("file", {}).get("url", "")
            else:
                url = ""
            
            # Get caption if available
            caption_parts = pdf_data.get("caption", [])
            caption = "".join([t.get("plain_text", "") for t in caption_parts])
            
            if caption:
                return f"[PDF: {caption}]({url})"
            else:
                return f"[PDF]({url})"
        
        elif block_type == "bookmark":
            bookmark_data = block.get("bookmark", {})
            url = bookmark_data.get("url", "")
            
            # Get caption if available
            caption_parts = bookmark_data.get("caption", [])
            caption = "".join([t.get("plain_text", "") for t in caption_parts])
            
            if caption:
                return f"[BOOKMARK: {caption}]({url})"
            else:
                return f"[BOOKMARK]({url})"
        
        return ""
    
    def fetch_page_content(self, page_id: str) -> str:
        """
        Fetch the actual content (blocks) from a Notion page
        
        Args:
            page_id (str): Page ID
            
        Returns:
            str: Concatenated text content from all blocks
        """
        content_parts = []
        
        try:
            has_more = True
            start_cursor = None
            block_count = 0
            
            while has_more:
                url = f"https://api.notion.com/v1/blocks/{page_id}/children"
                params = {"page_size": 100}
                if start_cursor:
                    params["start_cursor"] = start_cursor
                
                response = requests.get(url, headers=HEADERS, params=params)
                response.raise_for_status()
                
                data = response.json()
                blocks = data.get("results", [])
                
                for block in blocks:
                    text = self._extract_block_text(block)
                    if text:
                        content_parts.append(text)
                        block_count += 1
                
                has_more = data.get("has_more", False)
                start_cursor = data.get("next_cursor")
            
            full_content = "\n\n".join(content_parts)
            if block_count > 0:
                print(f"    Fetched {block_count} content blocks")
            return full_content
            
        except requests.exceptions.RequestException as e:
            print(f"    Warning: Could not fetch content for page {page_id}: {e}")
            return ""
    
    def build_graph(self, pages: List[Dict]) -> nx.DiGraph:
        """
        Build knowledge graph from Notion pages
        
        Args:
            pages (List[Dict]): List of page objects from Notion
            
        Returns:
            nx.DiGraph: Knowledge graph
        """
        print("Building knowledge graph...")
        
        # First pass: Create all nodes
        for page in pages:
            page_id = page["id"]
            properties = page.get("properties", {})
            
            # Extract key properties
            node_data = {
                "id": page_id,
                "url": page.get("url"),
                "created_time": page.get("created_time"),
                "last_edited_time": page.get("last_edited_time"),
            }
            
            # Extract all properties
            for prop_name, prop_value in properties.items():
                extracted_value = self.extract_property_value(prop_value)
                node_data[prop_name] = extracted_value
            
            # Determine node type from hierarchy
            node_type = node_data.get("Type", "Unknown")
            node_data["node_type"] = node_type
            
            # Add node to graph
            name = node_data.get("Name", f"Node-{page_id[:8]}")
            self.graph.add_node(page_id, name=name, **node_data)
            
            print(f"  Added node: {name} (Type: {node_type})")
        
        # Second pass: Create edges for parent-child relationships
        print("\nCreating parent-child relationships...")
        for page in pages:
            page_id = page["id"]
            properties = page.get("properties", {})
            
            # Get parent relationships
            parent_prop = properties.get("Parent item", {})
            parent_ids = self.extract_property_value(parent_prop)
            
            if parent_ids and isinstance(parent_ids, list):
                for parent_id in parent_ids:
                    if parent_id in self.graph:
                        self.graph.add_edge(
                            parent_id, 
                            page_id, 
                            relationship="PARENT_OF"
                        )
                        
                        parent_name = self.graph.nodes[parent_id].get("name", "Unknown")
                        child_name = self.graph.nodes[page_id].get("name", "Unknown")
                        print(f"  {parent_name} → {child_name}")
        
        # Third pass: Add KB article nodes and relationships
        print("\nProcessing KB Articles...")
        kb_count = 0
        
        for page_id in list(self.graph.nodes()):
            node_data = self.graph.nodes[page_id]
            kb_article_ids = node_data.get("KB Articles", [])
            
            if kb_article_ids and isinstance(kb_article_ids, list):
                component_name = node_data.get("name", "Unknown")
                
                for kb_id in kb_article_ids:
                    # Fetch KB article metadata
                    kb_metadata = self.fetch_kb_article_metadata(kb_id)
                    
                    if kb_metadata:
                        kb_node_id = f"kb_{kb_id}"
                        kb_title = kb_metadata.get("title", "Untitled KB Article")
                        
                        # Add KB article as a node
                        self.graph.add_node(
                            kb_node_id,
                            name=kb_title,
                            node_type="KB_ARTICLE",
                            original_id=kb_id,
                            **kb_metadata
                        )
                        
                        # Create relationship from component to KB article
                        self.graph.add_edge(
                            page_id,
                            kb_node_id,
                            relationship="HAS_KB_ARTICLE"
                        )
                        
                        kb_count += 1
                        print(f"  {component_name} → KB: {kb_title}")
        
        # Fourth pass: Create alias relationships
        print("\nCreating alias relationships...")
        alias_map = {}  # Map alias to list of node IDs
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            aliases = node_data.get("Aliases", "")
            
            if aliases and isinstance(aliases, str):
                # Split aliases by comma
                alias_list = [a.strip() for a in aliases.split(",") if a.strip()]
                
                for alias in alias_list:
                    alias_lower = alias.lower()
                    if alias_lower not in alias_map:
                        alias_map[alias_lower] = []
                    alias_map[alias_lower].append(node_id)
        
        # Create edges between nodes with shared aliases
        for alias, node_ids in alias_map.items():
            if len(node_ids) > 1:
                # Create edges between all pairs
                for i, node1 in enumerate(node_ids):
                    for node2 in node_ids[i+1:]:
                        self.graph.add_edge(
                            node1,
                            node2,
                            relationship="ALIAS_OF",
                            alias=alias
                        )
                        
                        name1 = self.graph.nodes[node1].get("name", "Unknown")
                        name2 = self.graph.nodes[node2].get("name", "Unknown")
                        print(f"  {name1} ⟷ {name2} (alias: {alias})")
        
        print(f"\n✓ Graph built successfully!")
        print(f"  Total nodes: {self.graph.number_of_nodes()}")
        print(f"  Total edges: {self.graph.number_of_edges()}")
        print(f"  KB Articles: {kb_count}")
        
        return self.graph
    
    def node_to_document(self, node_id: str) -> Document:
        """
        Convert a graph node to a LangChain Document object.
        
        Following LangChain best practices:
        - page_content: Combined text from name, descriptions, and importance fields
        - metadata: Graph structure info (relationships, edges, URLs, etc.)
        - id: Node identifier
        
        Args:
            node_id (str): Node ID in the graph
            
        Returns:
            Document: LangChain Document object
            
        Raises:
            ImportError: If langchain_core is not installed
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain_core is required for Document conversion. "
                "Install it with: pip install langchain-core"
            )
        node_data = self.graph.nodes[node_id]
        
        # Build page_content from relevant text fields
        content_parts = []
        
        # Add name/title
        name = node_data.get("name", "")
        if name:
            content_parts.append(f"Name: {name}")
        
        # Add final description (preferred) or regular description
        final_desc = node_data.get("Final Description (DO NOT TOUCH)", "")
        if not final_desc:
            final_desc = node_data.get("Description (DO NOT CHANGE)", "")
        if not final_desc:
            final_desc = node_data.get("Description", "")
        
        if final_desc:
            content_parts.append(f"\nDescription: {final_desc}")
        
        # Add importance/component importance
        importance = node_data.get("Importance", "")
        if not importance:
            importance = node_data.get("Component's Importance", "")
        if importance:
            content_parts.append(f"\nImportance: {importance}")
        
        # Add example if available
        example = node_data.get("Example")
        if example and example != "":
            content_parts.append(f"\nExample: {example}")
        
        # Add KB article content if this is a KB article node
        if node_data.get("node_type") == "KB_ARTICLE":
            kb_content = node_data.get("page_content", "")
            if kb_content:
                content_parts.append(f"\nContent:\n{kb_content}")
        
        # Combine all content
        page_content = "\n".join(content_parts) if content_parts else name or "No content available"
        
        # Build metadata with graph structure
        metadata = {
            # Basic node info
            "node_id": node_id,
            "node_type": node_data.get("node_type", "Unknown"),
            "url": node_data.get("url", ""),
            "created_time": node_data.get("created_time", ""),
            "last_edited_time": node_data.get("last_edited_time", ""),
            
            # Graph relationships
            "parent_ids": [],
            "child_ids": [],
            "kb_article_ids": [],
            "aliases": [],
            
            # Additional properties (excluding already used text fields)
            "type": node_data.get("Type"),
        }
        
        # Get parent relationships
        parents = [p for p, c in self.graph.edges() if c == node_id 
                  and self.graph.edges[p, c].get("relationship") == "PARENT_OF"]
        metadata["parent_ids"] = ",".join(parents) if parents else ""
        
        # Get child relationships
        children = [c for p, c in self.graph.edges() if p == node_id 
                   and self.graph.edges[p, c].get("relationship") == "PARENT_OF"]
        metadata["child_ids"] = ",".join(children) if children else ""
        
        # Get KB article relationships
        kb_articles = [c for p, c in self.graph.edges() if p == node_id 
                     and self.graph.edges[p, c].get("relationship") == "HAS_KB_ARTICLE"]
        metadata["kb_article_ids"] = ",".join(kb_articles) if kb_articles else ""
        
        # Get hierarchical path
        hierarchical_path = self.get_hierarchical_path(node_id)
        metadata["hierarchical_path"] = " -> ".join(hierarchical_path)
        
        # Get aliases
        aliases = node_data.get("Aliases", "")
        if aliases and isinstance(aliases, str):
            metadata["aliases"] = aliases
        elif aliases and isinstance(aliases, list):
             metadata["aliases"] = ",".join(str(a) for a in aliases)
        else:
            metadata["aliases"] = ""
        
        # Add all other node properties to metadata (excluding text fields used in page_content)
        excluded_fields = {
            "name", "Final Description (DO NOT TOUCH)", "Description (DO NOT CHANGE)", 
            "Description", "Importance", "Component's Importance", "Example", "page_content"
        }
        for key, value in node_data.items():
            if key not in excluded_fields and key not in metadata:
                # Only include simple types supported by ChromaDB
                if isinstance(value, (str, int, float, bool)) or value is None:
                    metadata[key] = value
                elif isinstance(value, list):
                    # Convert lists to comma-separated strings
                    try:
                        metadata[key] = ",".join(str(v) for v in value)
                    except:
                        metadata[key] = str(value)
                else:
                    # Convert everything else to string
                    metadata[key] = str(value)
        
        # Create Document with id
        return Document(
            page_content=page_content,
            metadata=metadata,
            id=node_id
        )
    
    def to_langchain_documents(self) -> List[Document]:
        """
        Convert all graph nodes to LangChain Document objects.
        
        Returns:
            List[Document]: List of Document objects ready for ingestion
        """
        documents = []
        
        print("Converting graph nodes to LangChain Documents...")
        
        for node_id in self.graph.nodes():
            try:
                doc = self.node_to_document(node_id)
                documents.append(doc)
            except Exception as e:
                print(f"  Warning: Could not convert node {node_id}: {e}")
                continue
        
        print(f"✓ Converted {len(documents)} nodes to Documents")
        return documents
    
    def ingest_to_chromadb(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: str = CHROMA_PERSIST_DIR,
        batch_size: int = 100
    ):
        """
        Ingest all graph nodes as documents into ChromaDB vector store (LangChain 1.0).
        
        Args:
            collection_name (str): Name of the ChromaDB collection
            persist_directory (str): Directory to persist the vector store
            batch_size (int): Number of documents to process per batch
            
        Returns:
            Chroma: The initialized vector store instance
            
        Raises:
            ImportError: If required packages are not installed
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB integration requires: pip install langchain-chroma langchain-openai"
            )
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "Document conversion requires: pip install langchain-core"
            )
        
        print("\n" + "="*80)
        print("INGESTING TO CHROMADB VECTOR STORE")
        print("="*80)
        
        # Initialize OpenAI embeddings (LangChain 1.0 API)
        print(f"\nInitializing OpenAI embeddings (model: {EMBEDDING_MODEL})...")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Initialize ChromaDB vector store (LangChain 1.0 API)
        print(f"Initializing ChromaDB vector store...")
        print(f"  Collection: {collection_name}")
        print(f"  Persist directory: {persist_directory}")
        
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # Convert graph nodes to documents
        documents = self.to_langchain_documents()
        
        if not documents:
            print("⚠ No documents to ingest!")
            return vector_store
        
        # Ingest documents in batches
        total_docs = len(documents)
        print(f"\nIngesting {total_docs} documents in batches of {batch_size}...")
        
        all_ids = []
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
            
            try:
                # Add documents to vector store (LangChain 1.0 API)
                ids = vector_store.add_documents(documents=batch)
                all_ids.extend(ids)
                print(f"    ✓ Batch {batch_num} ingested successfully")
            except Exception as e:
                print(f"    ✗ Error ingesting batch {batch_num}: {e}")
                continue
        
        print(f"\n✓ Successfully ingested {len(all_ids)} documents to ChromaDB")
        print(f"  Vector store persisted to: {persist_directory}")
        
        return vector_store
    
    def query_vector_store(
        self,
        query: str,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: str = CHROMA_PERSIST_DIR,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Query the ChromaDB vector store for similar documents (LangChain 1.0).
        
        Args:
            query (str): Query string to search for
            collection_name (str): Name of the ChromaDB collection
            persist_directory (str): Directory where vector store is persisted
            k (int): Number of results to return
            filter_dict (Optional[Dict]): Metadata filter (e.g., {"node_type": "Feature"})
            
        Returns:
            List[Document]: List of similar documents with metadata
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB integration requires: pip install langchain-chroma langchain-openai"
            )
        
        # Initialize embeddings (LangChain 1.0 API)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Load existing vector store (LangChain 1.0 API)
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # Perform similarity search (LangChain 1.0 API)
        if filter_dict:
            results = vector_store.similarity_search(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = vector_store.similarity_search(query, k=k)
        
        return results
    
    def query_vector_store_with_scores(
        self,
        query: str,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: str = CHROMA_PERSIST_DIR,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Query ChromaDB and return documents with similarity scores (LangChain 1.0).
        
        Args:
            query (str): Query string to search for
            collection_name (str): Name of the ChromaDB collection
            persist_directory (str): Directory where vector store is persisted
            k (int): Number of results to return
            filter_dict (Optional[Dict]): Metadata filter
            
        Returns:
            List[Tuple[Document, float]]: List of (document, score) tuples
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB integration requires: pip install langchain-chroma langchain-openai"
            )
        
        # Initialize embeddings (LangChain 1.0 API)
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Load existing vector store (LangChain 1.0 API)
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        
        # Perform similarity search with scores (LangChain 1.0 API)
        if filter_dict:
            results = vector_store.similarity_search_with_score(
                query,
                k=k,
                filter=filter_dict
            )
        else:
            results = vector_store.similarity_search_with_score(query, k=k)
        
        return results
    
    def get_hierarchical_path(self, node_id: str) -> List[str]:
        """
        Get the full hierarchical path from root to this node
        
        Args:
            node_id (str): Node ID
            
        Returns:
            List[str]: Path as list of node names
        """
        path = []
        current = node_id
        visited = set()
        
        while current:
            if current in visited:
                break  # Circular reference
            visited.add(current)
            
            node_data = self.graph.nodes[current]
            path.insert(0, node_data.get("name", "Unknown"))
            
            # Find parent
            parents = [p for p, c in self.graph.edges() if c == current]
            current = parents[0] if parents else None
        
        return path
    
    def get_node_by_name(self, name: str) -> Optional[str]:
        """
        Find node ID by name (case-insensitive)
        
        Args:
            name (str): Node name to search for
            
        Returns:
            Optional[str]: Node ID or None
        """
        name_lower = name.lower()
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("name", "").lower() == name_lower:
                return node_id
        return None
    
    def get_children(self, node_id: str) -> List[str]:
        """
        Get all direct children of a node
        
        Args:
            node_id (str): Node ID
            
        Returns:
            List[str]: List of child node IDs
        """
        return list(self.graph.successors(node_id))
    
    def get_descendants(self, node_id: str) -> List[str]:
        """
        Get all descendants of a node (recursively)
        
        Args:
            node_id (str): Node ID
            
        Returns:
            List[str]: List of descendant node IDs
        """
        descendants = []
        
        def traverse(current_id):
            children = self.get_children(current_id)
            for child in children:
                if child not in descendants:
                    descendants.append(child)
                    traverse(child)
        
        traverse(node_id)
        return descendants
    
    def get_kb_articles(self, node_id: str) -> List[Dict]:
        """
        Get all KB articles attached to this node
        
        Args:
            node_id (str): Node ID
            
        Returns:
            List[Dict]: List of KB article node data
        """
        kb_articles = []
        
        for successor in self.graph.successors(node_id):
            node_data = self.graph.nodes[successor]
            if node_data.get("node_type") == "KB_ARTICLE":
                kb_articles.append(node_data)
        
        return kb_articles
    
    def save_graph(self, filename: str = "knowledge_graph.json"):
        """
        Save the knowledge graph to a JSON file
        
        Args:
            filename (str): Output filename
        """
        # Convert graph to JSON-serializable format
        graph_data = {
            "nodes": [],
            "edges": []
        }
        
        # Export nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node_dict = {"id": node_id}
            node_dict.update(node_data)
            graph_data["nodes"].append(node_dict)
        
        # Export edges
        for source, target, edge_data in self.graph.edges(data=True):
            edge_dict = {
                "source": source,
                "target": target
            }
            edge_dict.update(edge_data)
            graph_data["edges"].append(edge_dict)
        
        # Save to file
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Graph saved to {filename}")
    
    def save_graph_gexf(self, filename: str = "knowledge_graph.gexf"):
        """
        Save graph in GEXF format for visualization in Gephi
        
        Args:
            filename (str): Output filename
        """
        # Create a clean copy of the graph with only simple attributes
        clean_graph = nx.DiGraph()
        
        for node_id, node_data in self.graph.nodes(data=True):
            # Only include simple string/numeric attributes for GEXF
            clean_attrs = {
                "name": str(node_data.get("name", "Unknown")),
                "node_type": str(node_data.get("node_type", "Unknown")),
                "url": str(node_data.get("url", "")),
                "Type": str(node_data.get("Type", "")),
            }
            clean_graph.add_node(node_id, **clean_attrs)
        
        # Copy all edges
        for source, target, edge_data in self.graph.edges(data=True):
            clean_graph.add_edge(source, target, **edge_data)
        
        nx.write_gexf(clean_graph, filename)
        print(f"✓ Graph saved to {filename} (GEXF format for Gephi)")
    
    def print_graph_summary(self):
        """Print a summary of the knowledge graph"""
        print("\n" + "="*80)
        print("KNOWLEDGE GRAPH SUMMARY")
        print("="*80)
        
        # Count nodes by type
        node_types = {}
        for node_id, node_data in self.graph.nodes(data=True):
            node_type = node_data.get("node_type") or node_data.get("Type") or "Unknown"
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print("\nNode Types:")
        for node_type, count in sorted(node_types.items(), key=lambda x: (x[0] or "").lower()):
            print(f"  {node_type}: {count}")
        
        # Count edges by relationship
        edge_types = {}
        for source, target, edge_data in self.graph.edges(data=True):
            rel_type = edge_data.get("relationship", "Unknown")
            edge_types[rel_type] = edge_types.get(rel_type, 0) + 1
        
        print("\nRelationship Types:")
        for rel_type, count in sorted(edge_types.items(), key=lambda x: (x[0] or "").lower()):
            print(f"  {rel_type}: {count}")
        
        # Find root nodes (nodes with no parents)
        root_nodes = [
            node_id for node_id in self.graph.nodes()
            if self.graph.in_degree(node_id) == 0
            and self.graph.nodes[node_id].get("node_type") != "KB_ARTICLE"
        ]
        
        print(f"\nRoot Nodes (no parents): {len(root_nodes)}")
        for root_id in root_nodes[:10]:  # Show first 10
            root_data = self.graph.nodes[root_id]
            print(f"  - {root_data.get('name', 'Unknown')} ({root_data.get('node_type', 'Unknown')})")
        
        if len(root_nodes) > 10:
            print(f"  ... and {len(root_nodes) - 10} more")
        
        print("\n" + "="*80)
    
    def query_node(self, node_name: str) -> Optional[Dict]:
        """
        Query a node by name and return its full context
        
        Args:
            node_name (str): Name of the node to query
            
        Returns:
            Optional[Dict]: Node context with hierarchy and KB articles
        """
        node_id = self.get_node_by_name(node_name)
        
        if not node_id:
            return None
        
        node_data = dict(self.graph.nodes[node_id])
        
        # Add hierarchical context
        path = self.get_hierarchical_path(node_id)
        node_data["hierarchical_path"] = " → ".join(path)
        
        # Add children
        children = self.get_children(node_id)
        node_data["children"] = [
            self.graph.nodes[child].get("name", "Unknown")
            for child in children
        ]
        
        # Add KB articles
        kb_articles = self.get_kb_articles(node_id)
        node_data["kb_articles"] = [
            {
                "title": kb.get("name"),
                "url": kb.get("url"),
                "id": kb.get("original_id")
            }
            for kb in kb_articles
        ]
        
        return node_data


def main():
    """Main execution"""
    print("="*80)
    print("NOTION KNOWLEDGE GRAPH BUILDER")
    print("="*80 + "\n")
    
    # Initialize builder
    builder = NotionKnowledgeGraphBuilder(DATABASE_ID)
    
    # Fetch all pages from database
    pages = builder.fetch_database_pages()
    
    # Build knowledge graph
    graph = builder.build_graph(pages)
    
    # Print summary
    builder.print_graph_summary()
    
    # Save graph in multiple formats
    builder.save_graph("knowledge_graph.json")
    builder.save_graph_gexf("knowledge_graph.gexf")
    
    # Ingest to ChromaDB Vector Store (LangChain 1.0)
    try:
        vector_store = builder.ingest_to_chromadb(
            collection_name=CHROMA_COLLECTION_NAME,
            persist_directory=CHROMA_PERSIST_DIR,
            batch_size=100
        )
        
        # Test retrieval with example queries
        print("\n" + "="*80)
        print("TESTING VECTOR STORE RETRIEVAL")
        print("="*80)
        
        # Example 1: Basic semantic search
        print("\n📝 Example 1: Semantic search for 'PDF reports'")
        results = builder.query_vector_store("PDF reports", k=3)
        for i, doc in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Node: {doc.metadata.get('node_id', 'N/A')[:50]}")
            print(f"    Type: {doc.metadata.get('node_type')}")
            print(f"    Path: {doc.metadata.get('hierarchical_path', 'N/A')[:80]}...")
            print(f"    Content: {doc.page_content[:120]}...")
        
        # Example 2: Search with metadata filter
        print("\n\n📝 Example 2: Search for KB articles about 'dashboard'")
        results = builder.query_vector_store(
            "dashboard",
            k=3,
            filter_dict={"node_type": "KB_ARTICLE"}
        )
        print(f"  Found {len(results)} KB articles")
        for i, doc in enumerate(results, 1):
            print(f"\n  Result {i}:")
            print(f"    Type: {doc.metadata.get('node_type')}")
            print(f"    URL: {doc.metadata.get('url', 'N/A')[:80]}...")
            print(f"    Content: {doc.page_content[:120]}...")
        
        # Example 3: Search with similarity scores
        print("\n\n📝 Example 3: Search for 'leasing metrics' with scores")
        results_with_scores = builder.query_vector_store_with_scores("leasing metrics", k=3)
        for i, (doc, score) in enumerate(results_with_scores, 1):
            print(f"\n  Result {i} (similarity: {score:.4f}):")
            print(f"    Type: {doc.metadata.get('node_type')}")
            print(f"    Path: {doc.metadata.get('hierarchical_path', 'N/A')[:80]}...")
            print(f"    Content: {doc.page_content[:120]}...")
        
    except ImportError as e:
        print(f"\n⚠ Skipping ChromaDB ingestion: {e}")
        print("  To enable vector store features:")
        print("    pip install langchain-chroma langchain-openai")
    except Exception as e:
        print(f"\n✗ Error during ChromaDB operations: {e}")
        import traceback
        traceback.print_exc()
    
    # Example query
    print("\n" + "="*80)
    print("EXAMPLE QUERY")
    print("="*80)
    
    example_node = "First Touch Attribution"
    result = builder.query_node(example_node)
    
    if result:
        print(f"\nQuery: '{example_node}'")
        print(f"Path: {result.get('hierarchical_path')}")
        print(f"Type: {result.get('node_type')}")
        print(f"Description: {result.get('Final Description', 'N/A')[:200]}...")
        print(f"Children: {len(result.get('children', []))}")
        print(f"KB Articles: {len(result.get('kb_articles', []))}")
        
        if result.get('kb_articles'):
            print("\nKB Articles:")
            for kb in result['kb_articles']:
                print(f"  - {kb['title']}")
                print(f"    URL: {kb['url']}")
    else:
        print(f"\nNode '{example_node}' not found")
    
    print("\n" + "="*80)
    print("✓ Done!")
    print("="*80)


if __name__ == "__main__":
    main()

