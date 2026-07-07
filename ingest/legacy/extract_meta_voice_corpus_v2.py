#!/usr/bin/env python3
"""
Meta Data Export - Voice Corpus Extractor v2
Updated for 2026 Meta export folder structure.

Usage:
    python extract_meta_voice_corpus_v2.py ~/Downloads/facebook-export
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
YOUR_NAME = "Mitchell Williams"
YOUR_POSSIBLE_NAMES = [
    "Mitchell Williams",
    "mitchell williams", 
    "Mitchell",
    "mitchell",
    "mitwilli",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def decode_meta_text(text):
    """Fix Meta's weird encoding (latin-1 stored as UTF-8)."""
    if not text:
        return ""
    try:
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

def is_your_message(sender_name):
    """Check if the sender is you."""
    if not sender_name:
        return False
    sender_lower = sender_name.lower()
    return any(name.lower() in sender_lower or sender_lower in name.lower() 
               for name in YOUR_POSSIBLE_NAMES)

def timestamp_to_date(ts_ms):
    """Convert millisecond timestamp to ISO date string."""
    try:
        return datetime.fromtimestamp(ts_ms / 1000).isoformat()
    except:
        return None

def timestamp_to_year(ts_ms):
    """Convert millisecond timestamp to year."""
    try:
        return datetime.fromtimestamp(ts_ms / 1000).year
    except:
        return None

def unix_to_date(ts):
    """Convert unix timestamp (seconds) to ISO date string."""
    try:
        return datetime.fromtimestamp(ts).isoformat()
    except:
        return None

def unix_to_year(ts):
    """Convert unix timestamp (seconds) to year."""
    try:
        return datetime.fromtimestamp(ts).year
    except:
        return None

def load_json_file(filepath):
    """Safely load a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"    Warning: Could not load {filepath}: {e}")
        return None

# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================

def extract_messages(base_path):
    """Extract YOUR messages from Messenger conversations."""
    messages = []
    
    # Correct path for 2026 export structure
    inbox_path = base_path / "your_facebook_activity" / "messages" / "inbox"
    
    if not inbox_path.exists():
        print(f"    Messages inbox not found at {inbox_path}")
        return messages
    
    conversation_count = 0
    your_message_count = 0
    
    for conv_folder in inbox_path.iterdir():
        if not conv_folder.is_dir():
            continue
        
        conversation_count += 1
        conv_name = conv_folder.name
        
        # Find all message JSON files
        for json_file in conv_folder.glob("message_*.json"):
            data = load_json_file(json_file)
            if not data or "messages" not in data:
                continue
            
            participants = [decode_meta_text(p.get("name", "")) 
                         for p in data.get("participants", [])]
            
            for msg in data["messages"]:
                sender = decode_meta_text(msg.get("sender_name", ""))
                
                # Only keep YOUR messages
                if not is_your_message(sender):
                    continue
                
                content = decode_meta_text(msg.get("content", ""))
                
                # Skip empty, reactions, or system messages
                if not content:
                    continue
                if content.startswith("You ") and "reaction" in content.lower():
                    continue
                if "sent an attachment" in content.lower():
                    continue
                if "started a call" in content.lower():
                    continue
                
                your_message_count += 1
                ts = msg.get("timestamp_ms")
                
                messages.append({
                    "source": "messenger",
                    "type": "message",
                    "date": timestamp_to_date(ts),
                    "year": timestamp_to_year(ts),
                    "conversation": conv_name,
                    "participants": participants,
                    "content": content,
                    "word_count": len(content.split()),
                })
    
    print(f"    Found {your_message_count:,} of your messages across {conversation_count:,} conversations")
    return messages

def extract_archived_messages(base_path):
    """Extract YOUR messages from archived threads."""
    messages = []
    
    archived_path = base_path / "your_facebook_activity" / "messages" / "archived_threads"
    
    if not archived_path.exists():
        return messages
    
    count = 0
    for conv_folder in archived_path.iterdir():
        if not conv_folder.is_dir():
            continue
        
        for json_file in conv_folder.glob("message_*.json"):
            data = load_json_file(json_file)
            if not data or "messages" not in data:
                continue
            
            participants = [decode_meta_text(p.get("name", "")) 
                         for p in data.get("participants", [])]
            
            for msg in data["messages"]:
                sender = decode_meta_text(msg.get("sender_name", ""))
                
                if not is_your_message(sender):
                    continue
                
                content = decode_meta_text(msg.get("content", ""))
                if not content or "sent an attachment" in content.lower():
                    continue
                
                count += 1
                ts = msg.get("timestamp_ms")
                
                messages.append({
                    "source": "messenger_archived",
                    "type": "message",
                    "date": timestamp_to_date(ts),
                    "year": timestamp_to_year(ts),
                    "conversation": conv_folder.name,
                    "participants": participants,
                    "content": content,
                    "word_count": len(content.split()),
                })
    
    if count > 0:
        print(f"    Found {count:,} archived messages")
    return messages

def extract_posts(base_path):
    """Extract YOUR Facebook posts."""
    posts = []
    
    posts_dir = base_path / "your_facebook_activity" / "posts"
    
    if not posts_dir.exists():
        print(f"    Posts directory not found at {posts_dir}")
        return posts
    
    # Look for the posts files with underscores
    post_files = list(posts_dir.glob("your_posts*.json"))
    
    for posts_file in post_files:
        data = load_json_file(posts_file)
        if not data:
            continue
        
        # Handle list or dict format
        post_list = data if isinstance(data, list) else data.get("posts", [])
        
        for post in post_list:
            content = ""
            timestamp = post.get("timestamp")
            
            # Try to extract text from various locations
            if "data" in post:
                for item in post["data"]:
                    if "post" in item:
                        content = decode_meta_text(item["post"])
                        break
            
            if not content and "post" in post:
                content = decode_meta_text(post["post"])
            
            if not content:
                # Try title field
                content = decode_meta_text(post.get("title", ""))
            
            if not content:
                continue
            
            posts.append({
                "source": "facebook",
                "type": "post",
                "date": unix_to_date(timestamp),
                "year": unix_to_year(timestamp),
                "content": content,
                "word_count": len(content.split()),
            })
    
    print(f"    Found {len(posts):,} posts")
    return posts

def extract_comments(base_path):
    """Extract YOUR comments."""
    comments = []
    
    comments_dir = base_path / "your_facebook_activity" / "comments_and_reactions"
    
    if not comments_dir.exists():
        print(f"    Comments directory not found at {comments_dir}")
        return comments
    
    # Look for comments files
    comment_files = list(comments_dir.glob("comments*.json"))
    
    for comments_file in comment_files:
        data = load_json_file(comments_file)
        if not data:
            continue
        
        comment_list = data.get("comments_v2", data.get("comments", data if isinstance(data, list) else []))
        
        for comment in comment_list:
            content = ""
            timestamp = comment.get("timestamp")
            
            # Try various content locations
            if "data" in comment:
                for item in comment["data"]:
                    if "comment" in item:
                        comment_obj = item["comment"]
                        if isinstance(comment_obj, dict):
                            content = decode_meta_text(comment_obj.get("comment", ""))
                        else:
                            content = decode_meta_text(comment_obj)
                        break
            
            if not content and "comment" in comment:
                content = decode_meta_text(comment["comment"])
            
            if not content:
                continue
            
            comments.append({
                "source": "facebook",
                "type": "comment",
                "date": unix_to_date(timestamp),
                "year": unix_to_year(timestamp),
                "content": content,
                "word_count": len(content.split()),
            })
    
    print(f"    Found {len(comments):,} comments")
    return comments

def extract_stories(base_path):
    """Extract story text/captions if any."""
    stories = []
    
    stories_dir = base_path / "your_facebook_activity" / "stories"
    
    if not stories_dir.exists():
        return stories
    
    for json_file in stories_dir.glob("*.json"):
        data = load_json_file(json_file)
        if not data:
            continue
        
        story_list = data if isinstance(data, list) else data.get("stories", [])
        
        for story in story_list:
            content = decode_meta_text(story.get("title", story.get("text", "")))
            if not content:
                continue
            
            timestamp = story.get("timestamp")
            
            stories.append({
                "source": "facebook",
                "type": "story",
                "date": unix_to_date(timestamp),
                "year": unix_to_year(timestamp),
                "content": content,
                "word_count": len(content.split()),
            })
    
    if stories:
        print(f"    Found {len(stories):,} stories with text")
    return stories

# ============================================================
# MAIN
# ============================================================

def main(export_path):
    base_path = Path(export_path)
    
    if not base_path.exists():
        print(f"Error: Path does not exist: {export_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("META DATA EXPORT - VOICE CORPUS EXTRACTOR v2")
    print(f"{'='*60}")
    print(f"\nProcessing: {base_path}")
    print(f"Looking for content from: {YOUR_NAME}")
    
    all_content = []
    
    print("\n[1/5] Extracting Messenger messages (inbox)...")
    all_content.extend(extract_messages(base_path))
    
    print("\n[2/5] Extracting archived messages...")
    all_content.extend(extract_archived_messages(base_path))
    
    print("\n[3/5] Extracting Facebook posts...")
    all_content.extend(extract_posts(base_path))
    
    print("\n[4/5] Extracting comments...")
    all_content.extend(extract_comments(base_path))
    
    print("\n[5/5] Extracting stories...")
    all_content.extend(extract_stories(base_path))
    
    # Sort by date
    all_content.sort(key=lambda x: x.get("date") or "0000")
    
    # Statistics
    total_words = sum(item.get("word_count", 0) for item in all_content)
    by_source = defaultdict(int)
    by_type = defaultdict(int)
    by_year = defaultdict(int)
    
    for item in all_content:
        by_source[item["source"]] += 1
        by_type[item["type"]] += 1
        if item.get("year"):
            by_year[item["year"]] += 1
    
    output = {
        "extraction_info": {
            "source": "Meta/Facebook Data Export",
            "extracted_for": YOUR_NAME,
            "extracted_at": datetime.now().isoformat(),
            "original_path": str(base_path),
            "script_version": "2.0"
        },
        "statistics": {
            "total_items": len(all_content),
            "total_words": total_words,
            "by_source": dict(by_source),
            "by_type": dict(by_type),
            "by_year": dict(sorted(by_year.items())),
            "date_range": {
                "earliest": all_content[0].get("date") if all_content else None,
                "latest": all_content[-1].get("date") if all_content else None,
            }
        },
        "corpus": all_content,
    }
    
    # Write output
    output_filename = f"meta_voice_corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = base_path.parent / output_filename
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nTotal items: {len(all_content):,}")
    print(f"Total words: {total_words:,}")
    print(f"\nBy source:")
    for source, count in sorted(by_source.items()):
        print(f"  {source}: {count:,}")
    print(f"\nBy type:")
    for type_, count in sorted(by_type.items()):
        print(f"  {type_}: {count:,}")
    
    if by_year:
        print(f"\nYear range: {min(by_year.keys())} - {max(by_year.keys())}")
    
    output_size = output_path.stat().st_size
    print(f"\n{'='*60}")
    print(f"OUTPUT: {output_path}")
    print(f"SIZE: {output_size / (1024*1024):.2f} MB")
    print(f"{'='*60}")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_meta_voice_corpus_v2.py ~/Downloads/facebook-export")
        sys.exit(1)
    
    main(sys.argv[1])
