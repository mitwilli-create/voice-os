#!/usr/bin/env python3
"""
Meta Data Export - Voice Corpus Extractor
Extracts only YOUR text content from a Meta/Facebook data export.
Outputs a clean JSON file suitable for Voice OS analysis.

Usage:
    python extract_meta_voice_corpus.py /path/to/facebook-mitwilli-2026-01-19-JLdFdpj4

(Unzip the archive first, then point to the extracted folder)
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ============================================================
# CONFIGURATION - Update this with your name as it appears in Meta
# ============================================================
YOUR_NAME = "Mitchell Williams"  # Exact match to your name in the export
YOUR_POSSIBLE_NAMES = [
    "Mitchell Williams",
    "mitchell williams", 
    "Mitchell",
    "mitchell",
    "mitwilli",  # username variations
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def decode_meta_text(text):
    """
    Meta exports encode text weirdly - this fixes mojibake issues.
    They encode as latin-1 but it's actually UTF-8.
    """
    if not text:
        return ""
    try:
        # Meta's weird encoding fix
        return text.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text

def is_your_message(sender_name):
    """Check if the sender is you (case-insensitive)."""
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

def load_json_file(filepath):
    """Safely load a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {filepath}: {e}")
        return None

# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================

def extract_messages(base_path):
    """
    Extract YOUR messages from Messenger conversations.
    Returns list of message objects.
    """
    messages = []
    messages_path = base_path / "messages" / "inbox"
    
    if not messages_path.exists():
        # Try alternate paths
        alt_paths = [
            base_path / "messages" / "archived_threads",
            base_path / "messages" / "filtered_threads",
            base_path / "your_activity_across_facebook" / "messages" / "inbox",
        ]
        for alt in alt_paths:
            if alt.exists():
                messages_path = alt
                break
    
    if not messages_path.exists():
        print("  Messages folder not found")
        return messages
    
    conversation_count = 0
    your_message_count = 0
    
    # Each conversation is a folder
    for conv_folder in messages_path.iterdir():
        if not conv_folder.is_dir():
            continue
        
        conversation_count += 1
        conv_name = conv_folder.name
        
        # Find all message JSON files in this conversation
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
                
                # Skip empty messages, reactions-only, or system messages
                if not content or content.startswith("You ") and "reaction" in content.lower():
                    continue
                
                your_message_count += 1
                messages.append({
                    "source": "messenger",
                    "type": "message",
                    "date": timestamp_to_date(msg.get("timestamp_ms")),
                    "year": datetime.fromtimestamp(msg.get("timestamp_ms", 0) / 1000).year if msg.get("timestamp_ms") else None,
                    "conversation": conv_name,
                    "participants": participants,
                    "content": content,
                    "word_count": len(content.split()),
                })
    
    print(f"  Found {your_message_count} of your messages across {conversation_count} conversations")
    return messages

def extract_posts(base_path):
    """
    Extract YOUR Facebook posts.
    """
    posts = []
    
    # Try various possible paths
    possible_paths = [
        base_path / "posts" / "your_posts_1.json",
        base_path / "posts" / "your_posts.json",
        base_path / "your_activity_across_facebook" / "posts" / "your_posts_1.json",
        base_path / "your_posts_check_ins_photos_and_videos_1.json",
    ]
    
    # Also check for numbered files (your_posts_1.json, your_posts_2.json, etc.)
    posts_dir = base_path / "posts"
    if posts_dir.exists():
        possible_paths.extend(posts_dir.glob("your_posts*.json"))
    
    for posts_path in possible_paths:
        if not Path(posts_path).exists():
            continue
            
        data = load_json_file(posts_path)
        if not data:
            continue
        
        # Handle different formats
        post_list = data if isinstance(data, list) else data.get("posts", data.get("status_updates", []))
        
        for post in post_list:
            # Extract text from various possible locations
            content = ""
            
            if "data" in post:
                for item in post["data"]:
                    if "post" in item:
                        content = decode_meta_text(item["post"])
                        break
            
            if not content and "post" in post:
                content = decode_meta_text(post["post"])
            
            if not content:
                content = decode_meta_text(post.get("title", ""))
            
            if not content:
                continue
            
            timestamp = post.get("timestamp")
            
            posts.append({
                "source": "facebook",
                "type": "post",
                "date": datetime.fromtimestamp(timestamp).isoformat() if timestamp else None,
                "year": datetime.fromtimestamp(timestamp).year if timestamp else None,
                "content": content,
                "word_count": len(content.split()),
            })
    
    print(f"  Found {len(posts)} posts")
    return posts

def extract_comments(base_path):
    """
    Extract YOUR comments on posts.
    """
    comments = []
    
    possible_paths = [
        base_path / "comments" / "comments.json",
        base_path / "comments_and_reactions" / "comments.json",
        base_path / "your_activity_across_facebook" / "comments" / "comments.json",
    ]
    
    for comments_path in possible_paths:
        if not comments_path.exists():
            continue
            
        data = load_json_file(comments_path)
        if not data:
            continue
        
        comment_list = data if isinstance(data, list) else data.get("comments", [])
        
        for comment in comment_list:
            # Extract comment text
            content = ""
            
            if "data" in comment:
                for item in comment["data"]:
                    if "comment" in item:
                        content = decode_meta_text(item["comment"].get("comment", ""))
                        break
            
            if not content:
                content = decode_meta_text(comment.get("comment", ""))
            
            if not content:
                continue
            
            timestamp = comment.get("timestamp")
            
            comments.append({
                "source": "facebook",
                "type": "comment",
                "date": datetime.fromtimestamp(timestamp).isoformat() if timestamp else None,
                "year": datetime.fromtimestamp(timestamp).year if timestamp else None,
                "content": content,
                "word_count": len(content.split()),
            })
    
    print(f"  Found {len(comments)} comments")
    return comments

def extract_instagram_content(base_path):
    """
    Extract Instagram posts, comments, and stories text.
    """
    items = []
    
    # Instagram posts/captions
    ig_paths = [
        base_path / "instagram" / "content" / "posts_1.json",
        base_path / "instagram" / "posts_1.json",
        base_path / "content" / "posts_1.json",
    ]
    
    for ig_path in ig_paths:
        if not ig_path.exists():
            continue
            
        data = load_json_file(ig_path)
        if not data:
            continue
        
        post_list = data if isinstance(data, list) else data.get("posts", [])
        
        for post in post_list:
            caption = decode_meta_text(post.get("title", post.get("caption", "")))
            if not caption:
                continue
            
            timestamp = post.get("creation_timestamp", post.get("taken_at"))
            
            items.append({
                "source": "instagram",
                "type": "post",
                "date": datetime.fromtimestamp(timestamp).isoformat() if timestamp else None,
                "year": datetime.fromtimestamp(timestamp).year if timestamp else None,
                "content": caption,
                "word_count": len(caption.split()),
            })
    
    # Instagram comments
    ig_comments_paths = [
        base_path / "instagram" / "comments" / "post_comments.json",
        base_path / "comments" / "post_comments.json",
    ]
    
    for comments_path in ig_comments_paths:
        if not comments_path.exists():
            continue
            
        data = load_json_file(comments_path)
        if not data:
            continue
        
        for comment in data if isinstance(data, list) else data.get("comments", []):
            content = decode_meta_text(comment.get("text", comment.get("comment", "")))
            if not content:
                continue
            
            timestamp = comment.get("timestamp")
            
            items.append({
                "source": "instagram",
                "type": "comment",
                "date": datetime.fromtimestamp(timestamp).isoformat() if timestamp else None,
                "year": datetime.fromtimestamp(timestamp).year if timestamp else None,
                "content": content,
                "word_count": len(content.split()),
            })
    
    print(f"  Found {len(items)} Instagram items")
    return items

# ============================================================
# MAIN EXTRACTION
# ============================================================

def main(export_path):
    base_path = Path(export_path)
    
    if not base_path.exists():
        print(f"Error: Path does not exist: {export_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("META DATA EXPORT - VOICE CORPUS EXTRACTOR")
    print(f"{'='*60}")
    print(f"\nProcessing: {base_path}")
    print(f"Looking for content from: {YOUR_NAME}")
    
    # Collect all content
    all_content = []
    
    print("\n[1/4] Extracting Messenger messages...")
    all_content.extend(extract_messages(base_path))
    
    print("\n[2/4] Extracting Facebook posts...")
    all_content.extend(extract_posts(base_path))
    
    print("\n[3/4] Extracting Facebook comments...")
    all_content.extend(extract_comments(base_path))
    
    print("\n[4/4] Extracting Instagram content...")
    all_content.extend(extract_instagram_content(base_path))
    
    # Sort by date
    all_content.sort(key=lambda x: x.get("date") or "0000")
    
    # Calculate statistics
    total_words = sum(item.get("word_count", 0) for item in all_content)
    by_source = defaultdict(int)
    by_type = defaultdict(int)
    by_year = defaultdict(int)
    
    for item in all_content:
        by_source[item["source"]] += 1
        by_type[item["type"]] += 1
        if item.get("year"):
            by_year[item["year"]] += 1
    
    # Build output
    output = {
        "extraction_info": {
            "source": "Meta/Facebook Data Export",
            "extracted_for": YOUR_NAME,
            "extracted_at": datetime.now().isoformat(),
            "original_path": str(base_path),
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
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"\nTotal items extracted: {len(all_content):,}")
    print(f"Total words: {total_words:,}")
    print(f"\nBy source:")
    for source, count in sorted(by_source.items()):
        print(f"  - {source}: {count:,}")
    print(f"\nBy type:")
    for type_, count in sorted(by_type.items()):
        print(f"  - {type_}: {count:,}")
    print(f"\nYear range: {min(by_year.keys()) if by_year else 'N/A'} - {max(by_year.keys()) if by_year else 'N/A'}")
    
    # File size
    output_size = output_path.stat().st_size
    print(f"\n{'='*60}")
    print(f"OUTPUT FILE: {output_path}")
    print(f"SIZE: {output_size / (1024*1024):.2f} MB")
    print(f"{'='*60}")
    
    if output_size > 50 * 1024 * 1024:  # > 50MB
        print("\n⚠️  Output is still large. Consider uploading in chunks or")
        print("    re-running with date filters (edit the script).")
    else:
        print("\n✓ Output file is small enough to upload directly!")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_meta_voice_corpus.py /path/to/unzipped/facebook-export")
        print("\nSteps:")
        print("  1. Unzip your Meta export: facebook-mitwilli-2026-01-19-JLdFdpj4.zip")
        print("  2. Run this script pointing to the extracted folder")
        print("  3. Upload the resulting JSON file to Claude")
        sys.exit(1)
    
    main(sys.argv[1])
