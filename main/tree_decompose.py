import os
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import argparse
import json
import time
from typing import List, Tuple, Dict, Counter
from collections import Counter, defaultdict
from openai import OpenAI
from retrieve import answer_question, direct_answer

from config import  DATASET, METHOD, CHUNK_SIZE, MIN_SENTENCE, OVERLAP, TOPK1, TOPK2, MAX_ITERATIONS, BASE_URL, API_KEY, MODEL_NAME, TREES_PER_QUESTION, MAX_TOKENS,                        DECOMPOSE_TEMPERATURE, TOP_P,FREQUENCY_PENALTY, PRESENCE_PENALTY, NUM_EXAMPLES, MAX_HEIGHT, RIGHT_SUBTREE_VARIANTS, RIGHT_SUBTREE_TREES_PER_VARIANT, MAX_VARIANTS, STATS_FILE_PATH,ENHANCED_RIGHT_SUBTREE, QUESTION_TIMEOUT_SECONDS
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
import os
import datetime
from pathlib import Path

# Add this function to save tree statistics to a file
def save_tree_stats(question, answer, original_height, final_height, file_path, success=True):
    """
    Save tree statistics to a file
    
    Parameters:
    - question: The original question
    - answer: The final answer obtained
    - original_height: Initial height of the tree before expansion
    - final_height: Final height after expansion and solving
    - file_path: Path to save the stats
    - success: Whether the question was successfully answered
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Format current timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Clean up the question and answer for single-line storage
    clean_question = question.replace('\n', ' ').strip()
    clean_answer = answer.replace('\n', ' ').strip() if answer else "[none]"
    success_str = "SUCCESS" if success else "FAILURE"
    
    # Prepare the statistics line
    stats_line = f"{timestamp}|{success_str}|{original_height}|{final_height}|{clean_question}|{clean_answer}\n"
    
    # Append to file
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(stats_line)
    
    print(f"Tree statistics saved to {file_path}")

def generate_response(messages, max_tokens=800, temperature=0.2, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    """Generic API call function to replace original requests.post call"""
    try:#"Qwen/Qwen2.5-14B-Instruct"
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"API call error: {str(e)}")
        return None
def generate_right_question_with_llm(parent_question, left_question, left_answer, original_right_question, max_tokens=800, temperature=0.2, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    
   
    messages = [
        {
            "role": "system",
            "content": "You are an intelligent AI assistant tasked with generating appropriate follow-up questions based on context. Generate clear, relevant questions that flow naturally from previous information."
        },
        {
            "role": "user",
            "content": f"""Based on the following information, please generate an appropriate follow-up question:

Original main question: {parent_question}
First sub-question: {left_question}
Answer to the first sub-question: {left_answer}
Original second sub-question (may contain replacement markers): {original_right_question}

Please generate a new second sub-question that:
1. Preserves all the essential information from the original second sub-question
2. Reads naturally and flows well in the conversation context
3. Replaces any markers like [answer_subquestion1] with natural language

Provide only the new question text without any additional explanation."""
        }
    ]
    
   
    response = generate_response(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    
   
    new_question = response.strip() if response else "Unable to generate a follow-up question."
    
 
    if "[answer_subquestion1]" in new_question or "[answer from" in new_question:
        new_question = new_question.replace("[answer_subquestion1]", "the aforementioned")
        new_question = new_question.replace("[answer from", "the aforementioned")
    
    return new_question

#---------------------------------------------------- Question Variant Generation -----------------------------------------------
def generate_question_variants(original_query, num_variants=2):
    """
    Generate question variants that maintain the exact semantics of the original query
    to ensure all variants would receive the same answer as the original question

    Parameters:
    original_query (str): The original query
    history_queries (list, optional): List of historical queries to avoid duplicating
    variant_type (str, optional): Type of variant to generate (e.g., 'wh_transform', 'syntax_reform', 'random')
    num_variants (int, optional): Number of unique variants to generate

    Returns:
    list: List of generated question variants that preserve exact semantics
    """
    system_prompt = """
      You are an expert at rewriting questions. Your job is to generate simpler, clearer versions of a question
    without changing its meaning or the answer it would receive.

    CRITICAL RULE: You must carefully analyze the original question to identify ANY specific constraints on the answer type.
    If the original question asks about a specific type of entity (like a city, date, person, number, etc.), 
    ALL your rewrites MUST explicitly preserve that exact entity type constraint.

    Each question you generate must:
    1. Preserve the original semantics exactly.
    2. NEVER replace a specific entity type with a more general one.
    3. Be simpler and more concise in sentence structure.
    4. Avoid unnecessary modifiers or extra descriptive phrases while retaining all semantic constraints.
    5. Use natural and direct English.

    Only return the list of revised questions, nothing else.
    """
    
    # Build user prompt with emphasis on simplification and syntax clarity
    user_prompt = f"""
    Original query: {original_query}

    
    Instructions:
    - First, identify any specific entity type being requested in the original query
    - If the original specifies a particular entity type, your rewrite MUST explicitly include that same type.
    - Do NOT generalize specific entity types under any circumstances.
    - Rewrite the question to be simpler and clearer while preserving ALL constraints.
    - All rewrites must have the EXACT SAME answer constraints as the original.
    - Do not repeat or include the original question in the outputs.
    - Return {num_variants} cleanly formatted variants, each on a new line and numbered.

    Start:
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response_text = generate_response(messages, temperature=0.6)
    
    # Parse the numbered list of variants
    variants = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('- ')):
            # Remove the numbering or bullet points
            variant = line.split('.', 1)[-1].strip() if '.' in line else line[2:].strip()
            variants.append(variant)
    
    # If no proper format is detected, treat the whole response as one variant
    if not variants and response_text.strip():
        variants = [response_text.strip()]
    
    # Return only the requested number of variants
    variants = variants[:num_variants]
    
    # Add the original query to the list of variants
    all_variants = [original_query] + variants
    
    print(f"Generated {len(variants)} question variants:")
    for i, variant in enumerate(all_variants):
        print(f"{i}. {variant}")
    
    return all_variants

def generate_right_question_with_llm(parent_question, left_question, left_answer, original_right_question, max_tokens=800, temperature=0.1, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    messages = [
        {
            "role": "system",
            "content": "You are a precise reasoning system that carefully analyzes question semantics."
        },
        {
            "role": "user",
            "content": f"""Original main question: {parent_question}
First sub-question: {left_question}
Answer to first sub-question: {left_answer}
Original second sub-question template: {original_right_question}

FIRST: Analyze whether the parent question is asking about:
- A SINGLE entity (e.g., "the person who...", "his/her")
- MULTIPLE entities (e.g., "the inventors...", "scientists who...", requires collaboration)

THEN: Generate the second sub-question following these core rules:

1. If parent question refers to A SINGLE ENTITY but answer has multiple entities:
   - Use "or" between entities (e.g., "A or B")
   - Adjust pronouns accordingly (e.g., "their" instead of "his")

2. If parent question refers to MULTIPLE ENTITIES:
   - Keep "and" between entities (e.g., "A and B")
   - Keep plural pronouns

3. Always preserve relationship structures 
Generate ONLY the replacement question:"""
        }
    ]
    
    response = generate_response(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    
    new_question = response.strip() if response else "Unable to generate a follow-up question."
    return new_question


def build_enhanced_right_subtree(original_question, left_answer, api_url=None, max_tokens=800, 
                            temperature=0.2, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0, 
                            examples_db=None, num_examples=3, max_height=3,
                            num_variants=2, trees_per_variant=3):
    """
Build multiple trees for the right subtree of Sequential type, following the same logic as the main function



Parameters:

-original_question: Original right subtree problem

-left_answer: The answer of the left subtree, used to replace [answer_subquestion1]

Other parameters are the same as build_question_tree



Returns:

-best_node: The best right subtree node
    """
    global global_node_counter
    

    if "[answer_subquestion1]" in original_question:
        question = original_question.replace("[answer_subquestion1]", left_answer)
    else:
        question = original_question
    
    print(f"\n{'='*80}")
    print(f"Construct the right subtree based on the problem: '{question}'")
    print(f"{'='*80}")
    
   
    max_attempts = num_variants
    current_attempt = 0
    current_question = question
    attempted_questions = [question]  
    
    while current_attempt < max_attempts:
        print(f"\nTry #{current_attempt+1} use the question: '{current_question}'")
        
        all_trees = []
        trees_to_generate = trees_per_variant  
        
        print(f"\nGenerate a {trees_to_generate} tree (maximum height: {max_height}) for the current problem.")
        
        for j in range(trees_to_generate):
         
            tree_temp = temperature
            
            print(f"\n #{j+1} (temperature={tree_temp}, max_height={max_height}):")
            
           
            root = build_question_tree(
                current_question, api_url, max_tokens, tree_temp, top_p, frequency_penalty,
                presence_penalty, examples_db, num_examples, depth=0,
                placeholder_answers={}, max_height=max_height
            )
            
            
            height, node_count = get_tree_statistics(root)
            
           
            all_trees.append({
                'tree': root,
                'tree_num': j+1,
                'height': height,
                'node_count': node_count,
                'question_text': current_question
            })
            
            
        
       
        tree_shape_counter = Counter()
        for tree_info in all_trees:
            shape = (tree_info['height'], tree_info['node_count'])
            tree_shape_counter[shape] += 1
        
      
        print("\nTree shape frequency (height, number of nodes)")
        for shape, count in tree_shape_counter.most_common():
            print(f"height: {shape [0]}, the number of nodes: {shape [1]} - frequency: {count}")
        

        if tree_shape_counter:
            most_common_shape, _ = tree_shape_counter.most_common(1)[0]
            print(f"\nMost common shapes: Height {most_common_shape[0]}, Number of nodes {most_common_shape[1]}")
            
           
            most_common_trees = [tree_info for tree_info in all_trees 
                               if (tree_info['height'], tree_info['node_count']) == most_common_shape]
            
            if most_common_trees:
                
                best_tree_info = most_common_trees[0]
                
                
                
                return best_tree_info['tree']
        
        
        current_attempt += 1
        if current_attempt < max_attempts:
            
            
            
            new_variants = generate_question_variants(question, num_variants=1)
            
            
            if len(new_variants) > 1:
                new_question = new_variants[1]
                
                current_question = new_question
                attempted_questions.append(current_question)
            else:
                print(f"Warning: Variant generation failed. Use the original question.")
                break 
    
   
    
    leaf_node = QuestionNode(
        question=question, 
        q_type="None",
        subq1=question, 
        subq2=""
    )
    return leaf_node




#---------------------------------------------------- Question Structure Analysis -----------------------------------------------
def analyze_question_structure(question, api_url=None):
    """
    Analyze question and return its structure with limiting conditions
    """
    examples_string = (
    "Question: \"Which female astronaut who graduated from Stanford University was the first to perform a spacewalk in the 1990s?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. The question asks for the identity of a specific astronaut with multiple defining characteristics.\"\n"
    "\"2. The astronaut is described as female, a Stanford University graduate, and the first to perform a spacewalk in the 1990s.\"\n"
    "\"3. For known entities, I need to identify explicit subjects mentioned in the question.\"\n"
    "\"4. Stanford University is explicitly named, and spacewalks in the 1990s are a specific time-limited event.\"\n"
    "\"5. The specific astronaut's identity is not directly provided - I need to determine who matches these criteria.\"\n"
    "\"6. Since I need to discover which specific person matches these characteristics, the astronaut identity is an unknown entity.\"\n"
    "So the structure is: [Core Query: Which astronaut Known Entities: {Subject: Stanford University, Limitation: educational institution}, {Subject: Spacewalk, Limitation: occurred in 1990s} Unknown Entities: {Subject: Astronaut identity, Limitation: female, Stanford graduate, first to perform spacewalk in 1990s}]\n\n"
    
    "Question: \"What disease affecting both livestock and humans was successfully eradicated worldwide by 1980 through a coordinated vaccination campaign?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. The question asks for a disease with specific characteristics and history.\"\n"
    "\"2. The disease must affect both livestock and humans, and was eradicated by 1980 through vaccination.\"\n"
    "\"3. For known entities, I need to identify explicit subjects mentioned in the question.\"\n"
    "\"4. Livestock and humans are explicitly mentioned as affected groups.\"\n"
    "\"5. The vaccination campaign and 1980 timeframe are explicitly mentioned parameters.\"\n"
    "\"6. The specific disease identity is what I need to discover, making it an unknown entity.\"\n"
    "So the structure is: [Core Query: What disease Known Entities: {Subject: Livestock, Limitation: affected by the disease}, {Subject: Humans, Limitation: affected by the disease}, {Subject: Vaccination campaign, Limitation: coordinated, completed by 1980, worldwide} Unknown Entities: {Subject: Disease identity, Limitation: affected both livestock and humans, eradicated by 1980}]\n\n"
    
    "Question: \"What architectural style is shared by the buildings designed by the same architect who constructed the famous cathedral located in Barcelona?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. The question seeks an architectural style shared across buildings.\"\n"
    "\"2. These buildings were designed by the architect who constructed a famous cathedral in Barcelona.\"\n"
    "\"3. For known entities, only Barcelona and the cathedral are explicitly mentioned.\"\n"
    "\"4. I need to discover multiple unknown pieces of information in sequence.\"\n"
    "\"5. First, I need to identify who the architect of the Barcelona cathedral was.\"\n"
    "\"6. Then, I need to identify other buildings designed by this architect.\"\n"
    "\"7. Finally, I need to determine what architectural style these buildings share.\"\n"
    "\"8. Each of these represents a distinct unknown entity in my analysis.\"\n"
    "So the structure is: [Core Query: What architectural style is shared Known Entities: {Subject: Cathedral, Limitation: famous, located in Barcelona}, {Subject: Barcelona, Limitation: city containing the cathedral} Unknown Entities: {Subject: Architect identity, Limitation: constructed Barcelona cathedral}, {Subject: Other buildings, Limitation: designed by same architect}, {Subject: Architectural style, Limitation: common to these buildings}]\n\n"
    
    "Question: \"What is the capital of the country where the inventor of dynamite was born?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. This question asks for a capital city through multiple steps of reasoning.\"\n"
    "\"2. The only explicitly named entity is dynamite, which is a known entity.\"\n"
    "\"3. I need to discover three distinct pieces of information to answer this question.\"\n"
    "\"4. First, I need to identify who invented dynamite - this person is not explicitly named.\"\n"
    "\"5. Then, I need to determine which country this person was born in.\"\n"
    "\"6. Finally, I need to identify the capital of that country.\"\n"
    "\"7. Each of these represents a separate unknown entity that requires factual knowledge.\"\n"
    "So the structure is: [Core Query: What is the capital Known Entities: {Subject: Dynamite, Limitation: explosive invention} Unknown Entities: {Subject: Inventor identity, Limitation: person who created dynamite}, {Subject: Country of birth, Limitation: birthplace of identified inventor}, {Subject: Capital city, Limitation: capital of identified country}]\n\n"
    
    
    "Question: \"Who was the teacher of the philosopher who taught Alexander the Great?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. This question asks for a teacher's identity through a chain of relationships.\"\n"
    "\"2. The only explicitly named entity is Alexander the Great, a historical figure.\"\n"
    "\"3. The question describes 'the philosopher who taught Alexander the Great' - this philosopher's identity is not given.\"\n"
    "\"4. Since I need to determine who this philosopher was, their identity is an unknown entity.\"\n"
    "\"5. Once I identify the philosopher, I need to determine who taught them - another unknown entity.\"\n"
    "\"6. This is a classic multi-step question requiring the identification of two distinct unknown entities.\"\n"
    "So the structure is: [Core Query: Who was the teacher Known Entities: {Subject: Alexander the Great, Limitation: historical figure} Unknown Entities: {Subject: Philosopher identity, Limitation: taught Alexander the Great}, {Subject: Teacher identity, Limitation: taught the identified philosopher}]\n\n"
    
    "Question: \"What musical technique is characteristic of compositions by the teacher of the pianist who performed at the opening ceremony of the 1980 Moscow Olympics?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. The question seeks a musical technique characteristic of certain compositions.\"\n"
    "\"2. The explicitly named entities are the 1980 Moscow Olympics and its opening ceremony.\"\n"
    "\"3. The question refers to 'the pianist who performed' - this pianist's identity is not given.\"\n"
    "\"4. It also refers to 'the teacher of the pianist' - this teacher's identity is also not given.\"\n"
    "\"5. I need to discover three distinct pieces of information to answer this question.\"\n"
    "\"6. First, I need to identify which pianist performed at the specified Olympic ceremony.\"\n"
    "\"7. Then, I need to identify who taught this pianist.\"\n"
    "\"8. Finally, I need to determine what musical technique characterizes the teacher's compositions.\"\n"
    "\"9. Each of these represents a separate unknown entity that requires factual knowledge.\"\n"
    "So the structure is: [Core Query: What musical technique is characteristic Known Entities: {Subject: Olympic ceremony, Limitation: opening, held in Moscow in 1980} Unknown Entities: {Subject: Pianist identity, Limitation: performed at specified Olympic ceremony}, {Subject: Teacher identity, Limitation: taught the identified pianist}, {Subject: Musical technique, Limitation: characteristic of the teacher's compositions}]\n\n"
    
    "Question: \"What scientific discovery was made by the mentor of the researcher who identified the double helix structure of DNA?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. The question asks for a scientific discovery through a chain of relationships.\"\n"
    "\"2. The explicitly named entity is DNA with its double helix structure specification.\"\n"
    "\"3. The question refers to 'the researcher who identified' - this researcher's identity is not given.\"\n"
    "\"4. It also refers to 'the mentor of the researcher' - this mentor's identity is also not given.\"\n"
    "\"5. I need to discover three distinct pieces of information to answer this question.\"\n"
    "\"6. First, I need to identify who discovered the double helix structure of DNA.\"\n"
    "\"7. Then, I need to identify who mentored this researcher.\"\n"
    "\"8. Finally, I need to determine what scientific discovery the mentor made.\"\n"
    "\"9. Each of these represents a separate unknown entity that requires factual knowledge.\"\n"
    "So the structure is: [Core Query: What scientific discovery was made Known Entities: {Subject: DNA, Limitation: biological molecule with double helix structure} Unknown Entities: {Subject: Researcher identity, Limitation: identified the double helix structure of DNA}, {Subject: Mentor identity, Limitation: mentored the identified researcher}, {Subject: Scientific discovery, Limitation: made by the identified mentor}]\n\n"
    
    "Question: \"Who is Boraqchin (Wife Of Ögedei)'s father-in-law?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. The question asks for a scientific discovery through a chain of relationships.\"\n"
    "\"2. The explicitly named entities are Boraqchin and Ögedei, with Boraqchin specifically identified as Ögedei's wife.\"\n"
    "\"3. Since Boraqchin is identified as Ögedei's wife, her father-in-law would logically be Ögedei's father.\"\n"
    "\"4. I need to determine who Ögedei's father was to identify Boraqchin's father-in-law.\"\n"
    "\"5. The core query is seeking the identity of a specific person (the father-in-law).\"\n"
    "\"6. The known entities are Boraqchin (with the limitation that she is Ögedei's wife) and Ögedei himself.\"\n"
    "\"7. The question requires sequential reasoning: first identifying Ögedei's father, then understanding this person is Boraqchin's father-in-law.\"\n"
    "\"8. This family relationship chain is central: spouse's father is father-in-law.\"\n"
    "So the structure is: [Core Query: Who is person's father-in-law Known Entities: {Subject: Boraqchin, Limitation: Wife of Ögedei}, {Subject: Ögedei, Limitation: Boraqchin's husband} Unknown Entities: {Subject: Father-in-law identity, Limitation: father of Ögedei, spouse's father to Boraqchin}, {Subject: Family relationship chain, Limitation: spouse relationship connects Boraqchin to Ögedei's father}]\n\n"

    "Question: \"What literary movement influenced the author who wrote the novel featuring a character who lives on Baker Street and solves mysteries using deductive reasoning?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. The question asks about a literary movement through a chain of relationships.\"\n"
    "\"2. The explicitly named entity is Baker Street (a location).\"\n"
    "\"3. The question describes a character with specific traits, but doesn't name them directly.\"\n"
    "\"4. It also refers to 'the author who wrote' - this author's identity is not given.\"\n"
    "\"5. I need to discover multiple distinct pieces of information to answer this question.\"\n"
    "\"6. First, I need to identify which character lives on Baker Street and solves mysteries using deduction.\"\n"
    "\"7. Then, I need to identify which author created this character.\"\n"
    "\"8. Finally, I need to determine what literary movement influenced this author.\"\n"
    "\"9. Each of these represents a separate unknown entity that requires factual knowledge.\"\n"
    "So the structure is: [Core Query: What literary movement influenced Known Entities: {Subject: Baker Street, Limitation: fictional residence location}, {Subject: Deductive reasoning, Limitation: method used to solve mysteries} Unknown Entities: {Subject: Character identity, Limitation: lives on Baker Street, uses deductive reasoning}, {Subject: Author identity, Limitation: created the identified character}, {Subject: Literary movement, Limitation: influenced the identified author}]\n\n"
    
    "Question: \"What painting technique was pioneered by the artist who created the most expensive artwork sold at auction in the same decade that the Berlin Wall fell?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. The question asks about a painting technique through a chain of relationships.\"\n"
    "\"2. The explicitly named entity is the Berlin Wall, with its fall as a historical event.\"\n"
    "\"3. The question refers to 'the artist who created' - this artist's identity is not given.\"\n"
    "\"4. It also refers to 'the most expensive artwork' - this artwork's identity is not given.\"\n"
    "\"5. I need to discover multiple distinct pieces of information to answer this question.\"\n"
    "\"6. First, I need to identify when the Berlin Wall fell and what decade that was.\"\n"
    "\"7. Then, I need to identify which artwork was the most expensive sold at auction in that decade.\"\n"
    "\"8. Next, I need to identify who created that artwork.\"\n"
    "\"9. Finally, I need to determine what painting technique this artist pioneered.\"\n"
    "\"10. Each of these represents a separate unknown entity that requires factual knowledge.\"\n"
    "So the structure is: [Core Query: What painting technique was pioneered Known Entities: {Subject: Berlin Wall, Limitation: historical structure that fell} Unknown Entities: {Subject: Decade, Limitation: when Berlin Wall fell}, {Subject: Artwork, Limitation: most expensive sold at auction in identified decade}, {Subject: Artist identity, Limitation: created the identified artwork}, {Subject: Painting technique, Limitation: pioneered by the identified artist}]\n\n"
    
    "Question: \"What philosophical concept was central to the teachings of the professor who mentored the author of the most influential paper on artificial intelligence published in the 1950s?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. The question asks about a philosophical concept through a chain of relationships.\"\n"
    "\"2. The explicitly named entities are artificial intelligence (field) and the 1950s (time period).\"\n"
    "\"3. The question refers to 'the author of the most influential paper' - this author's identity is not given.\"\n"
    "\"4. It also refers to 'the professor who mentored the author' - this professor's identity is not given.\"\n"
    "\"5. I need to discover multiple distinct pieces of information to answer this question.\"\n"
    "\"6. First, I need to identify which paper on AI from the 1950s was most influential.\"\n"
    "\"7. Then, I need to identify who authored this paper.\"\n"
    "\"8. Next, I need to identify who mentored this author.\"\n"
    "\"9. Finally, I need to determine what philosophical concept was central to this mentor's teachings.\"\n"
    "\"10. Each of these represents a separate unknown entity that requires factual knowledge.\"\n"
    "So the structure is: [Core Query: What philosophical concept was central Known Entities: {Subject: Artificial intelligence, Limitation: academic field}, {Subject: 1950s, Limitation: time period} Unknown Entities: {Subject: Paper identity, Limitation: most influential on AI, published in 1950s}, {Subject: Author identity, Limitation: wrote the identified paper}, {Subject: Professor identity, Limitation: mentored the identified author}, {Subject: Philosophical concept, Limitation: central to the identified professor's teachings}]\n\n"
    
    "Question: \"Which city was the birthplace of both Albert Einstein and Max Planck?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. This question asks about a city that is the birthplace of two specific people.\"\n"
    "\"2. The explicitly named entities are Albert Einstein and Max Planck, both historical scientists.\"\n"
    "\"3. The question contains the logical 'both...and' construction, indicating that the city must satisfy two conditions simultaneously.\"\n"
    "\"4. I need to discover two distinct pieces of information and check if they match.\"\n"
    "\"5. First, I need to identify where Albert Einstein was born.\"\n"
    "\"6. Then, I need to identify where Max Planck was born.\"\n"
    "\"7. Finally, I need to determine if these are the same city.\"\n"
    "So the structure is: [Core Query: Which city Known Entities: {Subject: Albert Einstein, Limitation: historical scientist}, {Subject: Max Planck, Limitation: historical scientist} Unknown Entities: {Subject: Einstein's birthplace, Limitation: city where Albert Einstein was born}, {Subject: Planck's birthplace, Limitation: city where Max Planck was born}, {Subject: Common birthplace, Limitation: city that satisfies both conditions if it exists}]\n\n"

    "Question: \"What is the capital of France or Italy?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. This question asks about the capital city of either France or Italy.\"\n"
    "\"2. The explicitly named entities are France and Italy, both countries.\"\n"
    "\"3. The question contains a logical 'or' that creates two distinct possibilities.\"\n"
    "\"4. Each country's capital represents a separate unknown entity we might need to identify.\"\n"
    "\"5. The core query is asking for capital identification, but we need to clarify which country's capital is being requested.\"\n"
    "So the structure is: [Core Query: What is the capital Known Entities: {Subject: France, Limitation: country}, {Subject: Italy, Limitation: country} Unknown Entities: {Subject: France's capital, Limitation: capital city of France}, {Subject: Italy's capital, Limitation: capital city of Italy}]\n\n"

    "Question: \"The chemical element discovered by Marie Curie is used in which medical procedure?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. This question asks about a medical procedure using a specific chemical element.\"\n"
    "\"2. The explicitly named entity is Marie Curie, a historical scientist.\"\n"
    "\"3. The question refers to 'the chemical element discovered by Marie Curie' - this element is not named.\"\n"
    "\"4. I need to discover two distinct pieces of information to answer this question.\"\n"
    "\"5. First, I need to identify which chemical element was discovered by Marie Curie.\"\n"
    "\"6. Then, I need to identify which medical procedure uses this element.\"\n"
    "\"7. Each of these represents a separate unknown entity that requires factual knowledge.\"\n"
    "So the structure is: [Core Query: Which medical procedure Known Entities: {Subject: Marie Curie, Limitation: historical scientist} Unknown Entities: {Subject: Chemical element, Limitation: discovered by Marie Curie}, {Subject: Medical procedure, Limitation: uses the identified chemical element}]\n\n"
    
    "Question: \"The country bordered by the most nations is located on which continent?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. This question asks about the continent where a specific country is located.\"\n"
    "\"2. There are no explicitly named entities in this question.\"\n"
    "\"3. The question refers to 'the country bordered by the most nations' - this country is not named.\"\n"
    "\"4. I need to discover two distinct pieces of information to answer this question.\"\n"
    "\"5. First, I need to identify which country shares borders with the most other countries.\"\n"
    "\"6. Then, I need to determine which continent this country is located on.\"\n"
    "\"7. Each of these represents a separate unknown entity that requires factual knowledge.\"\n"
    "So the structure is: [Core Query: Which continent Known Entities: {} Unknown Entities: {Subject: Country identity, Limitation: bordered by the most nations}, {Subject: Continent identity, Limitation: contains the identified country}]\n\n"
    
    "Question: \"The director of the movie that won Best Picture in 2010 was born in which city?\"\n"
    "CoT: Let's think step by step\n"
    "\"1. This question asks about a birthplace through a chain of relationships.\"\n"
    "\"2. The explicitly named entity is the year 2010, a time period.\"\n"
    "\"3. The question refers to 'the movie that won Best Picture in 2010' - this movie is not named.\"\n"
    "\"4. It also refers to 'the director of the movie' - this director is not named.\"\n"
    "\"5. I need to discover three distinct pieces of information to answer this question.\"\n"
    "\"6. First, I need to identify which movie won Best Picture in 2010.\"\n"
    "\"7. Then, I need to identify who directed this movie.\"\n"
    "\"8. Finally, I need to determine which city this director was born in.\"\n"
    "\"9. Each of these represents a separate unknown entity that requires factual knowledge.\"\n"
    "So the structure is: [Core Query: Which city Known Entities: {Subject: Best Picture award, Limitation: given in 2010}, {Subject: 2010, Limitation: specific year} Unknown Entities: {Subject: Movie identity, Limitation: won Best Picture in 2010}, {Subject: Director identity, Limitation: directed the identified movie}, {Subject: Birthplace, Limitation: city where the identified director was born}]\n\n"
    
)
    
    system_prompt = (
        "You will analyze questions by breaking them down into their components. For each question, your response MUST strictly follow this format:\n\n"
        "1. Begin with 'CoT: Let's think step by step'\n"
        "2. Number your reasoning steps as \"1.\", \"2.\", etc., each in quotes\n"
        "3. After your reasoning, write 'So the structure is:' followed by the structured breakdown\n\n"
        
        "The structure breakdown should contain these three components:\n"
        "- **Core Query**: The primary information being sought.\n"
        "- **Known Entities**: Information explicitly provided in the question, structured as {Subject: Entity, Limitation: time/space/other constraints}.\n"
        "- **Unknown Entities**: Information needed to answer the question, including intermediate steps in multi-hop questions, structured in the same format as Known Entities.\n\n"
        
        "Key principles:\n"
        "- Use consistent formatting: {Subject: Entity, Limitation: constraints}\n"
        "- Group subjects with their limitations in both sections\n"
        "- Include time periods, locations, and other qualifiers as limitations\n"
        "- Identify ALL unknown entities needed, including intermediate steps in multi-hop questions\n"
        "- Distinguish between explicitly mentioned (known) and implied/needed (unknown) information\n"
        "- Be precise about which limitations apply to which subjects\n"
        "- Ensure entities don't appear in both known and unknown categories\n\n"
        
        "The EXACT format for your final output must be:\n"
        "CoT: Let's think step by step\n"
        "\"1. [reasoning step]\"\n"
        "\"2. [reasoning step]\"\n"
        "... more reasoning steps as needed ...\n"
        "So the structure is: [Core Query: ... Known Entities: {Subject: Entity, Limitation: constraints}, {Subject: Another Entity, Limitation: constraints} Unknown Entities: {Subject: Entity, Limitation: constraints}, {Subject: Another Entity, Limitation: constraints}]"
    )
    
    user_message = examples_string + "\nQuestion: \"" + question + "\"\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    api_response = generate_response(messages, max_tokens=800, temperature=0)
    
    if api_response:
        # Replace multiple consecutive newlines with a single newline for easier processing
        api_response = api_response.replace("\n\n", "\n")
        
        structure_start = api_response.find("So the structure is:")
        if structure_start != -1:
            structure_text = api_response[structure_start + len("So the structure is:"):].strip()
            return structure_text
    
    return "[Core Query: Unknown Known Entities: Unknown Unknown Entities: Unknown]"




#---------------------------------------------------- Examples Database -----------------------------------------------
def get_examples_database():
    examples = [
        
        {
            "question": "What is the capital of France or Italy?",
            "structure": "[Core Query: What is the capital Known Entities: {Subject: France, Limitation: country}, {Subject: Italy, Limitation: country} Unknown Entities: {Subject: Capital of France, Limitation: city serving as French capital}, {Subject: Capital of Italy, Limitation: city serving as Italian capital}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'What is the capital', but the question contains the logical operator 'OR' connecting two countries: France and Italy. The known entities are the countries France and Italy. Looking at the unknown entities, I need to identify the capital city of each country. The logical 'OR' suggests the user wants information about either one or both capitals. I can deliberately IGNORE other cities in these countries or government structures - while these would provide context about the countries, they don't help identify their capitals. This requires parallel decomposition to independently determine each capital before presenting the information about both possibilities.",
            "subq1": "What is the capital of France?",
            "subq2": "What is the capital of Italy?"
        },
        {
            "question": "What language is spoken in both Switzerland and Luxembourg?",
            "structure": "[Core Query: What language is spoken in both Known Entities: {Subject: Switzerland, Limitation: country}, {Subject: Luxembourg, Limitation: country} Unknown Entities: {Subject: Languages of Switzerland, Limitation: officially spoken in Switzerland}, {Subject: Languages of Luxembourg, Limitation: officially spoken in Luxembourg}, {Subject: Common languages, Limitation: spoken in both countries}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query asks 'What language is spoken in both', seeking languages common to two countries. The known entities are Switzerland and Luxembourg. The logical 'BOTH' indicates I need to find the intersection of two sets. Looking at the unknown entities, I need to identify languages spoken in Switzerland and languages spoken in Luxembourg, then determine which appear in both sets. I can deliberately IGNORE dialect variations or historical language development - while linguistically interesting, they don't help identify which official languages are shared between the countries. This requires parallel decomposition to independently determine each set of languages before finding their intersection.",
            "subq1": "What languages are spoken in Switzerland?",
            "subq2": "What languages are spoken in Luxembourg?"
        },
        {
    "question": "Which actor starred in The Godfather and Apocalypse Now?",
    "structure": "[Core Query: Which actor starred Known Entities: {Subject: The Godfather, Limitation: film}, {Subject: Apocalypse Now, Limitation: film} Unknown Entities: {Subject: Godfather cast, Limitation: actors who starred in The Godfather}, {Subject: Apocalypse Now cast, Limitation: actors who starred in Apocalypse Now}, {Subject: Common actors, Limitation: appeared in both films}]",
    "type": "Parallel",
    "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query asks 'Which actor starred', seeking performers common to two films. The known entities are The Godfather and Apocalypse Now. The logical 'AND' indicates I need to find the intersection of two sets. Looking at the unknown entities, I need to identify actors who starred in The Godfather and actors who starred in Apocalypse Now, then determine who appears in both casts. I can deliberately IGNORE directors, release dates, or plot details - while these provide context about the films, they don't help identify which actors appeared in both. This requires parallel decomposition to independently determine each cast list before finding their intersection.",
    "subq1": "Which actors starred in The Godfather?",
    "subq2": "Which actors starred in Apocalypse Now?"
},
        
        {
            "question": "When did the director of the film \"The Seventh Seal\" die?",
            "structure": "[Core Query: When did person die Known Entities: {Subject: Film, Limitation: \"The Seventh Seal\"} Unknown Entities: {Subject: Director identity, Limitation: of \"The Seventh Seal\"}, {Subject: Death date, Limitation: of identified director}, {Subject: Director career, Limitation: other works}, {Subject: Film details, Limitation: release date, critical reception}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I'll determine if decomposition is necessary. The core query is 'When did person die', seeking a death date. The known entity is the film 'The Seventh Seal'. Looking at the unknown entities, I need director identity and death date to answer the question. I must first identify the director before finding their death date, as that's a clear dependency. However, I can deliberately IGNORE the director's career and film details - while they provide context about the director and film, they're not essential for answering when the director died. This requires sequential decomposition to first identify the director, then find their death date.",
            "subq1": "Who directed the film \"The Seventh Seal\"?",
            "subq2": "When did [answer_subquestion1] die?"
        },
        {
    "question": "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
    "structure": "[Core Query: Who is person's father-in-law Known Entities: {Subject: Boraqchin, Limitation: Wife of Ögedei}, {Subject: Ögedei, Limitation: Boraqchin's husband} Unknown Entities: {Subject: Father-in-law identity, Limitation: father of Ögedei, spouse's father to Boraqchin}, {Subject: Family relationship chain, Limitation: spouse relationship connects Boraqchin to Ögedei's father}]",
    "type": "Sequential",
    "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'Who is person's father-in-law'. The known entities are 'Boraqchin' and 'Ögedei'. Looking at the unknown entities, I need to identify Boraqchin's father-in-law. Since Boraqchin is explicitly identified as Ögedei's wife, her father-in-law would be Ögedei's father. I can deliberately IGNORE historical context about these figures - while this might provide interesting background, it doesn't help identify the specific person who is Ögedei's father. This requires sequential decomposition because finding Boraqchin's father-in-law depends on first identifying who Boraqchin's spouse is, then determining who that person's father was.",
    "subq1": "Who is Boraqchin's spouse?",
    "subq2": "Who is [answer_subquestion1]'s father?"
},

        {
            "question": "Did the composer of \"Symphony No. 9\" (Choral Symphony) die after the painter of \"The Starry Night\"?",
            "structure": "[Core Query: Did person A die after person B Known Entities: {Subject: Symphony, Limitation: \"Symphony No. 9\" (Choral Symphony)}, {Subject: Painting, Limitation: \"The Starry Night\"} Unknown Entities: {Subject: Composer identity, Limitation: of Symphony No. 9}, {Subject: Painter identity, Limitation: of The Starry Night}, {Subject: Death date, Limitation: of identified composer}, {Subject: Death date, Limitation: of identified painter}, {Subject: Birth dates, Limitation: of both artists}, {Subject: Artistic styles, Limitation: of both artists}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query compares two death dates: 'Did person A die after person B'. Known entities are the works 'Symphony No. 9' and 'The Starry Night'. Examining the unknown entities, I need composer identity, painter identity, and their death dates to make the comparison. I can deliberately IGNORE birth dates and artistic styles - while these might be interesting for context, they don't contribute to determining who died after whom. This requires parallel decomposition since I need to find each death date independently before comparing them.",
            "subq1": "When did the composer of \"Symphony No. 9\" (Choral Symphony) die?",
            "subq2": "When did the painter of \"The Starry Night\" die?"
        },
        
        {
            "question": "Which instrument did the composer of \"Symphony No. 9\" (Choral Symphony) play as a child?",
            "structure": "[Core Query: Which instrument did person play as a child Known Entities: {Subject: Symphony, Limitation: \"Symphony No. 9\" (Choral Symphony)} Unknown Entities: {Subject: Composer identity, Limitation: of Symphony No. 9}, {Subject: Instrument, Limitation: played by identified composer during childhood}, {Subject: Musical education, Limitation: training of composer}, {Subject: Composition style, Limitation: characteristics of composer's work}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query asks 'Which instrument did person play as a child'. The known entity is 'Symphony No. 9'. Looking at the unknown entities, I need to identify the composer first, then determine what instrument they played as a child. I can deliberately IGNORE musical education and composition style - while relevant to the composer's development, they don't directly answer what specific instrument was played. This requires sequential decomposition because identifying the childhood instrument depends on first knowing who the composer was.",
            "subq1": "Who composed \"Symphony No. 9\" (Choral Symphony)?",
            "subq2": "Which instrument did [answer_subquestion1] play as a child?"
        },
        
        {
            "question": "In which calendar year did the literary figure responsible for penning the dystopian novel '1984' ultimately pass away?",
            "structure": "[Core Query: In which year did person pass away Known Entities: {Subject: Novel, Limitation: dystopian, titled '1984'} Unknown Entities: {Subject: Author identity, Limitation: of the novel '1984'}, {Subject: Death year, Limitation: of identified author}, {Subject: Other works, Limitation: by same author}, {Subject: Political views, Limitation: of author that influenced novel}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I'll assess if decomposition is necessary. The core query seeks a death year with 'In which year did person pass away'. The known entity is the novel '1984'. Looking at the unknown entities, I need to identify the author first, then determine their death year. I can deliberately IGNORE other works and political views - while these provide context about the author's career and influences, they don't help identify when the author died. This requires sequential decomposition since finding the death year depends on first identifying the author.",
            "subq1": "Who authored the dystopian novel '1984'?",
            "subq2": "In which year did [answer_subquestion1] pass away?"
        },
        
        {
            "question": "When did the director of film Hypocrite (Film) die?",
            "structure": "[Core Query: When did person die Known Entities: {Subject: Film, Limitation: titled \"Hypocrite\"} Unknown Entities: {Subject: Director identity, Limitation: of film Hypocrite}, {Subject: Death date, Limitation: of identified director}, {Subject: Film release date, Limitation: when Hypocrite was released}, {Subject: Director's filmography, Limitation: other films by same director}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'When did person die', asking for a death date. The known entity is the film 'Hypocrite'. Looking at the unknown entities, I need to identify the director before I can find their death date. I can deliberately IGNORE the film release date and director's filmography - while these provide context about the film and director's career, they don't help determine when the director died. This requires sequential decomposition because finding the death date depends on first identifying who directed the film.",
            "subq1": "Who directed the film Hypocrite (Film)?",
            "subq2": "When did [answer_subquestion1] die?"
        },
        
        {
            "question": "Are both Kurram Garhi and Trojkrsti located in the same country?",
            "structure": "[Core Query: Are both located in the same country Known Entities: {Subject: Kurram Garhi, Limitation: location name}, {Subject: Trojkrsti, Limitation: location name} Unknown Entities: {Subject: Country, Limitation: containing Kurram Garhi}, {Subject: Country, Limitation: containing Trojkrsti}, {Subject: Geographic features, Limitation: of both locations}, {Subject: Population data, Limitation: of both locations}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to assess if decomposition is necessary. The core query asks 'Are both located in the same country', a comparison question. Known entities are locations 'Kurram Garhi' and 'Trojkrsti'. Looking at the unknown entities, I need to determine the country for each location to make the comparison. I can deliberately IGNORE geographic features and population data - while these provide context about the locations, they don't help determine which countries they're in. This requires parallel decomposition to independently determine each location's country before comparing them.",
            "subq1": "In which country is Kurram Garhi located?",
            "subq2": "In which country is Trojkrsti located?"
        },
        
        {
            "question": "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            "structure": "[Core Query: Do person A and person B have the same nationality Known Entities: {Subject: Film A, Limitation: Coolie No. 1 (1995)}, {Subject: Film B, Limitation: The Sensational Trial} Unknown Entities: {Subject: Director A identity, Limitation: of Coolie No. 1}, {Subject: Director B identity, Limitation: of The Sensational Trial}, {Subject: Nationality, Limitation: of Director A}, {Subject: Nationality, Limitation: of Director B}, {Subject: Film genres, Limitation: of both films}, {Subject: Box office performance, Limitation: of both films}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query asks if two people share the same nationality. Known entities are the films 'Coolie No. 1' and 'The Sensational Trial'. Looking at the unknown entities, I need to identify each director and their nationalities to make the comparison. I can deliberately IGNORE film genres and box office performance - while these provide context about the films, they're irrelevant to the directors' nationalities. This requires parallel decomposition since I can find each director's nationality independently before comparing them.",
            "subq1": "What is the nationality of the director of Coolie No. 1 (1995 Film)?",
            "subq2": "What is the nationality of the director of The Sensational Trial?"
        },
        
       
        
        {
            "question": "Who was born first out of Martin Hodge and Ivania Martinich?",
            "structure": "[Core Query: Who was born first Known Entities: {Subject: Martin Hodge, Limitation: person for comparison}, {Subject: Ivania Martinich, Limitation: person for comparison} Unknown Entities: {Subject: Birth date, Limitation: of Martin Hodge}, {Subject: Birth date, Limitation: of Ivania Martinich}, {Subject: Professional accomplishments, Limitation: of both individuals}, {Subject: Nationality, Limitation: of both individuals}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'Who was born first', a comparison of birth dates. Known entities are 'Martin Hodge' and 'Ivania Martinich'. Looking at the unknown entities, I need the birth dates of both individuals to determine who was born first. I can deliberately IGNORE professional accomplishments and nationalities - while these might provide context about who these people are, they're irrelevant to determining birth order. This requires parallel decomposition to independently find each birth date before comparing them.",
            "subq1": "When was Martin Hodge born?",
            "subq2": "When was Ivania Martinich born?"
        },
        
        {
            "question": "When did the director of film Laughter In Hell die?",
            "structure": "[Core Query: When did person die Known Entities: {Subject: Film, Limitation: titled \"Laughter In Hell\"} Unknown Entities: {Subject: Director identity, Limitation: of film Laughter In Hell}, {Subject: Death date, Limitation: of identified director}, {Subject: Film production details, Limitation: studio, budget}, {Subject: Director's other films, Limitation: filmography}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to assess if decomposition is necessary. The core query is 'When did person die', asking for a death date. The known entity is the film 'Laughter In Hell'. Looking at the unknown entities, I need to identify the director before I can find their death date. I can deliberately IGNORE film production details and the director's other films - while these provide context about the film and director's career, they don't help determine when the director died. This requires sequential decomposition because finding the death date depends on first identifying who directed the film.",
            "subq1": "Who directed the film Laughter In Hell?",
            "subq2": "When did [answer_subquestion1] die?"
        },
        
        {
            "question": "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            "structure": "[Core Query: Which film's director died later Known Entities: {Subject: Film A, Limitation: The Gal Who Took the West}, {Subject: Film B, Limitation: Twenty Plus Two} Unknown Entities: {Subject: Director A identity, Limitation: of The Gal Who Took the West}, {Subject: Director B identity, Limitation: of Twenty Plus Two}, {Subject: Death date, Limitation: of Director A}, {Subject: Death date, Limitation: of Director B}, {Subject: Film release dates, Limitation: of both films}, {Subject: Critical reception, Limitation: of both films}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query is 'Which film's director died later', comparing death dates. Known entities are the films 'The Gal Who Took the West' and 'Twenty Plus Two'. Looking at the unknown entities, I need to identify each director and their death dates to determine who died later. I can deliberately IGNORE film release dates and critical reception - while these provide context about the films, they don't help determine when the directors died. This requires parallel decomposition to independently find when each director died before comparing the dates.",
            "subq1": "When did the director of The Gal Who Took the West die?",
            "subq2": "When did the director of Twenty Plus Two die?"
        },
        
        {
            "question": "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            "structure": "[Core Query: Who is person's grandchild Known Entities: {Subject: Krishna Shah, Limitation: Nepalese Royal} Unknown Entities: {Subject: Child identity, Limitation: of Krishna Shah}, {Subject: Grandchild identity, Limitation: child of Krishna Shah's child}, {Subject: Royal lineage, Limitation: historical significance}, {Subject: Reign dates, Limitation: period of authority}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'Who is person's grandchild'. The known entity is 'Krishna Shah'. Looking at the unknown entities, I need to identify Shah's children first, then their children (Shah's grandchildren). I can deliberately IGNORE royal lineage and reign dates - while these provide historical context about Shah's position, they don't help identify specific family relationships. This requires sequential decomposition because identifying grandchildren depends on first identifying children.",
            "subq1": "Who is the child of Krishna Shah (Nepalese Royal)?",
            "subq2": "Who is the child of [answer_subquestion1]?"
        },
        
        {
            "question": "What is the official currency of Brazil?",
            "structure": "[Core Query: What is the official currency Known Entities: {Subject: Brazil, Limitation: country} Unknown Entities: {Subject: Currency, Limitation: official for Brazil}, {Subject: Currency history, Limitation: previous currencies of Brazil}, {Subject: Exchange rate, Limitation: value relative to USD}, {Subject: Economic indicators, Limitation: inflation, GDP}]",
            "type": "None",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is truly necessary. The core query is 'What is the official currency'. The known entity is 'Brazil'. Looking at the unknown entities, I only need to identify Brazil's current official currency. I can deliberately IGNORE currency history, exchange rates, and economic indicators - while they provide context about Brazil's economy, they don't help identify what the current official currency is. This is a straightforward factual question that can be answered in one step without decomposition.",
            "subq1": "What is the official currency of Brazil?",
            "subq2": ""
        },
        
        {
            "question": "What city, where the creator of 'The Scream' spent most of his childhood, is now considered a major cultural center?",
            "structure": "[Core Query: What city is a cultural center Known Entities: {Subject: Artwork, Limitation: 'The Scream'} Unknown Entities: {Subject: Creator identity, Limitation: of 'The Scream'}, {Subject: Childhood city, Limitation: where identified creator spent most childhood}, {Subject: Cultural status, Limitation: of identified city in present day}, {Subject: Artist's technique, Limitation: painting style}, {Subject: Museum location, Limitation: where artwork is displayed}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'What city is a cultural center'. The known entity is 'The Scream'. Looking at the unknown entities, I need to identify the creator of 'The Scream', then determine where they spent most of their childhood. I can deliberately IGNORE the artist's technique and museum location - while these provide context about the artwork, they don't help identify the childhood city. This requires sequential decomposition because identifying the city depends on first identifying the artist.",
            "subq1": "Who created 'The Scream'?",
            "subq2": "In which city did [answer_subquestion1] spend most of his childhood?"
        },
        
        {
            "question": "In which century did the composer, whose opera premiered in the same year as the French Revolution began, die?",
            "structure": "[Core Query: In which century did person die Known Entities: {Subject: French Revolution, Limitation: historical event with specific beginning year} Unknown Entities: {Subject: Composer identity, Limitation: whose opera premiered same year as French Revolution began}, {Subject: Death century, Limitation: of identified composer}, {Subject: Opera details, Limitation: title and musical style}, {Subject: Composer's nationality, Limitation: country of origin}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query is 'In which century did person die'. The known entity is the 'French Revolution'. Looking at the unknown entities, I need to identify which composer had an opera premiere coinciding with the start of the French Revolution, then determine in which century they died. I can deliberately IGNORE opera details and composer's nationality - while these provide context about the composer, they don't help determine when they died. This requires sequential decomposition because finding the death century depends on first identifying the specific composer.",
            "subq1": "Which composer's opera premiered in the same year as the French Revolution began?",
            "subq2": "In which century did [answer_subquestion1] die?"
        },
        
        {
            "question": "Which language, spoken by the indigenous people who first inhabited the region where Silicon Valley is now located, has fewer native speakers today?",
            "structure": "[Core Query: Which language has fewer speakers Known Entities: {Subject: Silicon Valley, Limitation: geographic region with specific indigenous history} Unknown Entities: {Subject: Indigenous languages, Limitation: spoken by first inhabitants of Silicon Valley region}, {Subject: Speaker counts, Limitation: current number of native speakers for each identified language}, {Subject: Cultural traditions, Limitation: of indigenous groups}, {Subject: Historical territories, Limitation: exact boundaries of indigenous lands}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'Which language has fewer speakers'. The known entity is 'Silicon Valley'. Looking at the unknown entities, I first need to identify which indigenous languages were spoken in the Silicon Valley region, then compare their current speaker counts to determine which has fewer. I can deliberately IGNORE cultural traditions and historical territories - while these provide important context about the indigenous peoples, they don't help determine current speaker counts. This requires sequential decomposition because comparing speaker counts depends on first identifying the relevant languages.",
            "subq1": "What indigenous languages were spoken by the people who first inhabited the region where Silicon Valley is now located?",
            "subq2": "Which of [answer_subquestion1] has the fewest native speakers today?"
        },
        
        {
            "question": "Did the mathematician whose theorem is fundamental to modern calculus die before or after the astronomer who first proposed the heliocentric model?",
            "structure": "[Core Query: Did person A die before or after person B Known Entities: {Subject: Calculus theorem, Limitation: fundamental to modern calculus}, {Subject: Heliocentric model, Limitation: astronomical theory} Unknown Entities: {Subject: Mathematician identity, Limitation: created fundamental calculus theorem}, {Subject: Astronomer identity, Limitation: first proposed heliocentric model}, {Subject: Death date, Limitation: of identified mathematician}, {Subject: Death date, Limitation: of identified astronomer}, {Subject: Publications, Limitation: major works of both scientists}, {Subject: Academic positions, Limitation: institutions where they worked}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query is 'Did person A die before or after person B', comparing death dates. Known entities are 'calculus theorem' and 'heliocentric model'. Looking at the unknown entities, I need to identify both the mathematician and astronomer, then determine their death dates to make the comparison. I can deliberately IGNORE publications and academic positions - while these provide context about their careers, they don't help determine when they died. This requires parallel decomposition because I can find each death date independently before comparing them.",
            "subq1": "When did the mathematician whose theorem is fundamental to modern calculus die?",
            "subq2": "When did the astronomer who first proposed the heliocentric model die?"
        },
        
        {
            "question": "Which painting, created by an artist who studied under the same teacher as Leonardo da Vinci, is currently housed in the Louvre Museum?",
            "structure": "[Core Query: Which painting is in the Louvre Known Entities: {Subject: Leonardo da Vinci, Limitation: famous artist}, {Subject: Louvre Museum, Limitation: art institution} Unknown Entities: {Subject: Teacher, Limitation: of da Vinci}, {Subject: Other students, Limitation: studied under same teacher as da Vinci}, {Subject: Paintings, Limitation: created by identified students and housed in Louvre}, {Subject: Artistic techniques, Limitation: used by the artists}, {Subject: Historical period, Limitation: when artists were active}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'Which painting is in the Louvre'. Known entities are 'Leonardo da Vinci' and 'Louvre Museum'. Looking at the unknown entities, I need to identify da Vinci's teacher, then other students of this teacher, then paintings by these artists in the Louvre. I can deliberately IGNORE artistic techniques and historical periods - while these provide context about the artists, they don't help identify which paintings are in the Louvre. This requires sequential decomposition because finding the paintings depends on first identifying the relevant artists.",
            "subq1": "Which artists studied under the same teacher as Leonardo da Vinci?",
            "subq2": "Which paintings created by [answer_subquestion1] are housed in the Louvre Museum?"
        },
        
        {
            "question": "What is the capital of the country where the inventor of the telephone spent his final years?",
            "structure": "[Core Query: What is the capital of country Known Entities: {Subject: Telephone, Limitation: invention} Unknown Entities: {Subject: Inventor identity, Limitation: of telephone}, {Subject: Country, Limitation: where identified inventor spent final years}, {Subject: Capital, Limitation: of identified country}, {Subject: Invention date, Limitation: when telephone was created}, {Subject: Other inventions, Limitation: by same inventor}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to assess if decomposition is necessary. The core query is 'What is the capital of country'. The known entity is 'telephone'. Looking at the unknown entities, I need to identify the telephone inventor, then determine where they spent their final years, and finally identify that country's capital. I can deliberately IGNORE the invention date and other inventions - while these provide context about the inventor's achievements, they don't help identify where they spent their final years or that country's capital. This requires sequential decomposition because each step depends on the previous one.",
            "subq1": "In which country did the inventor of the telephone spend his final years?",
            "subq2": "What is the capital of [answer_subquestion1]?"
        },
        
        {
            "question": "Which musical instrument, played by the composer who wrote the most famous requiem while on his deathbed, was his primary instrument?",
            "structure": "[Core Query: Which instrument was primary Known Entities: {Subject: Requiem, Limitation: most famous, written on deathbed} Unknown Entities: {Subject: Composer identity, Limitation: wrote famous requiem on deathbed}, {Subject: Primary instrument, Limitation: of identified composer}, {Subject: Other compositions, Limitation: by same composer}, {Subject: Musical era, Limitation: period when composer was active}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query is 'Which instrument was primary'. The known entity is the 'famous requiem written on deathbed'. Looking at the unknown entities, I need to identify the composer who wrote this requiem, then determine their primary instrument. I can deliberately IGNORE other compositions and musical era - while these provide context about the composer's work and time period, they don't help identify their primary instrument. This requires sequential decomposition because identifying the instrument depends on first identifying the composer.",
            "subq1": "Which composer wrote the most famous requiem while on his deathbed?",
            "subq2": "What was [answer_subquestion1]'s primary musical instrument?"
        },
        
        {
            "question": "From which university did the physicist, whose equation unifies electricity and magnetism into a single theory, graduate?",
            "structure": "[Core Query: From which university did person graduate Known Entities: {Subject: Equation, Limitation: unifies electricity and magnetism} Unknown Entities: {Subject: Physicist identity, Limitation: created unifying equation}, {Subject: University, Limitation: where identified physicist graduated}, {Subject: Year of graduation, Limitation: when physicist completed studies}, {Subject: Other scientific contributions, Limitation: additional work by physicist}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'From which university did person graduate'. The known entity is 'equation about electricity and magnetism'. Looking at the unknown entities, I need to first identify which physicist created this unifying equation, then determine their alma mater. I can deliberately IGNORE graduation year and other scientific contributions - while these provide context about the physicist's education and career, they don't help identify which university they attended. This requires sequential decomposition because finding the university depends on first identifying the physicist.",
            "subq1": "Which physicist's equation unifies electricity and magnetism into a single theory?",
            "subq2": "From which university did [answer_subquestion1] graduate?"
        },
        
        {
            "question": "Which disease, that caused the death of the author whose novel depicted a young orphan in Victorian England, was most prevalent in European cities of that era?",
            "structure": "[Core Query: Which disease was most prevalent Known Entities: {Subject: Victorian England, Limitation: historical period}, {Subject: Novel theme, Limitation: young orphan} Unknown Entities: {Subject: Author identity, Limitation: wrote novel about orphan in Victorian England}, {Subject: Disease, Limitation: caused identified author's death and prevalent in European cities of that era}, {Subject: Author's other works, Limitation: bibliography}, {Subject: Medical treatments, Limitation: available in Victorian era}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to assess if decomposition is necessary. The core query is 'Which disease was most prevalent'. Known entities are 'Victorian England' and 'novel about orphan'. Looking at the unknown entities, I need to identify the author of novels about orphans in Victorian England, then determine what disease caused their death and was also prevalent in that era. I can deliberately IGNORE the author's other works and medical treatments - while these provide historical context, they don't help identify the specific disease that caused the author's death. This requires sequential decomposition because identifying the disease depends on first identifying the author.",
            "subq1": "Which author wrote a novel depicting a young orphan in Victorian England?",
            "subq2": "What disease, that caused the death of [answer_subquestion1], was most prevalent in European cities of the Victorian era?"
        },
        
        {
            "question": "In what year was the landmark building, designed by the architect who also created the famous glass pyramid, completed?",
            "structure": "[Core Query: In what year was building completed Known Entities: {Subject: Glass pyramid, Limitation: famous architectural work} Unknown Entities: {Subject: Architect identity, Limitation: created the glass pyramid}, {Subject: Landmark building, Limitation: designed by identified architect}, {Subject: Completion year, Limitation: of identified landmark building}, {Subject: Architectural style, Limitation: of both structures}, {Subject: Construction materials, Limitation: used in both structures}]",
            "type": "Sequential",
            "cot": "Let's think step by step. First, I need to determine if decomposition is necessary. The core query is 'In what year was building completed'. The known entity is 'glass pyramid'. Looking at the unknown entities, I need to identify the architect who created the glass pyramid, then their landmark building, and finally its completion year. I can deliberately IGNORE architectural style and construction materials - while these provide interesting context about the structures, they don't help identify the completion year. This requires sequential decomposition due to the dependencies between identifying the architect, the building, and then its completion year.",
            "subq1": "Which architect created the famous glass pyramid?",
            "subq2": "In what year was the landmark building designed by [answer_subquestion1] completed?"
        },

        
        {
            "question": "Are both Sagrada Familia and Notre-Dame Cathedral designated as UNESCO World Heritage sites?",
            "structure": "[Core Query: Are both designated as UNESCO World Heritage sites Known Entities: {Subject: Sagrada Familia, Limitation: architectural landmark}, {Subject: Notre-Dame Cathedral, Limitation: architectural landmark} Unknown Entities: {Subject: UNESCO status, Limitation: of Sagrada Familia}, {Subject: UNESCO status, Limitation: of Notre-Dame Cathedral}, {Subject: Construction history, Limitation: of both buildings}, {Subject: Architectural styles, Limitation: of both buildings}]",
            "type": "Parallel",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is necessary. The core query is 'Are both designated as UNESCO World Heritage sites', a status comparison. Known entities are 'Sagrada Familia' and 'Notre-Dame Cathedral'. Looking at the unknown entities, I need to determine the UNESCO World Heritage status of each site. I can deliberately IGNORE construction history and architectural styles - while these provide context about the buildings, they don't help determine their UNESCO status. This requires parallel decomposition to check each status independently before comparing them.",
            "subq1": "Is Sagrada Familia designated as a UNESCO World Heritage site?",
            "subq2": "Is Notre-Dame Cathedral designated as a UNESCO World Heritage site?"
        },
        
        {
            "question": "Who was the first female astronaut to travel to space?",
            "structure": "[Core Query: Who was the first female astronaut Known Entities: {Subject: Space travel, Limitation: accomplished by females} Unknown Entities: {Subject: Astronaut identity, Limitation: female, first to travel to space}, {Subject: Launch date, Limitation: of first female space mission}, {Subject: Spacecraft, Limitation: used for first female space mission}, {Subject: Mission duration, Limitation: length of first female space flight}]",
            "type": "None",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is truly necessary. The core query is 'Who was the first female astronaut'. The known entity is 'space travel'. Looking at the unknown entities, I only need to identify which female was the first to travel to space. I can deliberately IGNORE launch date, spacecraft, and mission duration - while these provide interesting context about the historic mission, they don't help identify who the first female astronaut was. This is a straightforward factual question answerable in one step without decomposition.",
            "subq1": "Who was the first female astronaut to travel to space?",
            "subq2": ""
        },
        

        {
            "question": "When did the CEO of Tesla also become the owner of the social media platform previously known as Twitter?",
            "structure": "[Core Query: When became owner Known Entities: {Subject: CEO of Tesla, Limitation: person with specific role}, {Subject: Social media platform, Limitation: previously known as Twitter} Unknown Entities: {Subject: Acquisition date, Limitation: when CEO became owner of platform}, {Subject: Purchase price, Limitation: amount paid for acquisition}, {Subject: Platform changes, Limitation: modifications after acquisition}]",
            "type": "None",
            "cot": "Let's think step by step. First, I need to evaluate if decomposition is truly necessary. The core query is 'When became owner', seeking a date. Known entities are 'CEO of Tesla' and 'social media platform previously known as Twitter'. Looking at the unknown entities, I only need the acquisition date to answer when the ownership changed. I can deliberately IGNORE purchase price and platform changes - while these provide context about the acquisition and its aftermath, they don't help determine when the ownership transfer occurred. This is a straightforward factual question answerable in one step without decomposition.",
            "subq1": "When did the CEO of Tesla also become the owner of the social media platform previously known as Twitter?",
            "subq2": ""
        },
        
    ]
    return examples


#---------------------------------------------------- Vector Similarity -----------------------------------------------
def find_similar_examples(question, examples, num_examples=3):
    example_questions = [ex["question"] for ex in examples]
    vectorizer = TfidfVectorizer()
    all_questions = example_questions + [question]
    vectors = vectorizer.fit_transform(all_questions)
    question_vector = vectors[-1]
    example_vectors = vectors[:-1]
    similarities = cosine_similarity(question_vector, example_vectors)[0]
    most_similar_indices = similarities.argsort()[-num_examples:][::-1]
    return [examples[i] for i in most_similar_indices]

#---------------------------------------------------- Prompt Construction -----------------------------------------------
def construct_prompt(question, examples, structure):
    prompt = """I want you to analyze questions and break them down into subproblems.
I'll provide the question and its structure analysis (Core Query, Known Entities, Limiting Conditions, and Unknown Entities).
Please analyze the question structure and determine how to decompose it. Follow this format:

    Question: [The original question]

Structure: [Analysis of core query, known entities, limiting conditions, and unknown entities]

CoT: Let's think step by step
[Detailed step-by-step reasoning examining the question structure]
[Analyze the core query, known entities, and limiting conditions]
[Evaluate which unknown entities are crucial for answering the question]
[Decide if decomposition is needed and what strategy to use]
[Ensure key limiting conditions are preserved in subquestions]

So the Type is: [Parallel, Sequential, or None]

So the Subquestion 1 is: [First subquestion; if type is None, should be identical to original question]

So the Subquestion 2 is: [Second subquestion; if Sequential, MUST include [answer_subquestion1]; if Parallel, MUST NOT reference subquestion 1; if None, leave empty]

**MANDATORY REQUIREMENTS:**

1. **DECOMPOSITION RULE**: Only break down into two subquestions when necessary.

2. **SIMPLICITY EVALUATION**: Questions answerable in one step should be classified as "None" type.

3. **SUBQUESTION CLARITY**: Each subquestion must yield a specific, factual, and unambiguous answer.

4. **NONE TYPE FORMATTING**: Subquestion 1 must match original question, Subquestion 2 must be empty.

5. **SEQUENTIAL REQUIREMENTS**: Subquestion 2 must contain [answer_subquestion1] placeholder and form a complete logical chain.

6. **PARALLEL REQUIREMENTS**: Both subquestions must be independent and together solve the original question.

7. **PRECISION**: Subquestions must include sufficient context to be answerable without clarification.

8. **TERMINOLOGY CONSISTENCY**: Always use "subquestion" consistently.

9. **ENTITY FOCUS**: Only include entities directly contributing to answering the core query.

10. **CONSISTENCY**: Your reasoning must align with your final decomposition.

11. **MEANINGFUL DECOMPOSITION**: Avoid trivial definitional subquestions.

12. **SUBSTANTIVE CONTRIBUTION**: Each subquestion must provide information that advances the solution.

13. **PRESERVING JOINT LIMITING CONDITIONS**: When multiple limiting conditions together define a single entity, they MUST be kept together and NEVER split across subquestions. Words like "both," "same," "together," "jointly," and similar terms signal that limiting conditions should be preserved as a unit. Questions like "Who invented both X and Y" or "Who is known for both A and B" should NOT be decomposed into parallel questions about separate entities.

14. **DECOMPOSITION VALIDATION**: Always verify the correctness and completeness of your decomposition before finalizing.


Here are some examples:
"""
    for ex in examples:
        prompt += f"""
    Question: {ex["question"]}

Structure: {ex["structure"]}

CoT: {ex["cot"]}

So the Type is: {ex["type"]}

So the Subquestion 1 is: {ex["subq1"]}"""
        if ex["type"] != "None":
            prompt += f"\nSo the Subquestion 2 is: {ex['subq2']}"
        prompt += "\n"
    prompt += f"""
### **Now, analyze the following question:**
    Question: {question}

Structure: {structure}

CoT: Let's think step by step
"""
    return prompt

#---------------------------------------------------- API Client -----------------------------------------------
def generate_responses(prompt, api_url=None, max_tokens=800, temperature=0, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
    """Using OpenAI client instead of original API call"""
    system_message = """You are an expert at analyzing questions and breaking them down into simpler subquestions.
You carefully distinguish between sequential questions (where the second subquestion depends on the answer to the first),
parallel questions (where both subquestions can be answered independently), and None-type questions (which are already simple).

**CRITICAL REQUIREMENTS YOU MUST ENFORCE:**

1. **DECOMPOSITION NECESSITY**: Only decompose a question when absolutely necessary. If a question can be answered directly, mark it as "None" type.

2. **SEQUENTIAL DECOMPOSITION CORRECTNESS**: For sequential types, VERIFY that substituting the answer from Subquestion 1 into Subquestion 2 directly yields the full answer to the original question. This is THE MOST IMPORTANT test of a valid sequential decomposition.

3. **NO TRIVIAL SUBQUESTIONS**: NEVER create basic definition questions like "Who/What is X?" unless absolutely necessary for intermediate reasoning AND not common knowledge.

4. **LOGICAL PATHWAY**: Ensure there is a clear, direct logical connection between subquestions and the original question. Every step must advance toward the final answer.

5. **ALIGNMENT BETWEEN COT AND DECOMPOSITION**: Your step-by-step reasoning must perfectly align with your final decomposition choice and subquestion formulation.

6. **VALIDATION CHECK**: Before finalizing, mentally substitute expected answers to verify your decomposition structure works.

7. **FOCUS ON DIRECT ANSWERABLE QUESTIONS**: Every subquestion must yield a specific, factual answer that directly contributes to solving the original question.

8. **PRESERVING JOINT LIMITING CONDITIONS**: It is ABSOLUTELY CRITICAL to never split multiple limiting conditions that together define a single entity. When a question asks about "both X and Y," "same," "together," "jointly," or similar terms indicating multiple attributes of ONE entity, DO NOT decompose these into separate questions. For example:

   - "Who invented  X and Y?" should NOT be split into "Who invented X?" and "Who invented Y?"
   - "Which country has both characteristic A and B?" must stay as a single question
   - "What is known for simultaneously doing X and Y?" must be kept intact

   This requirement takes precedence over other decomposition considerations. When multiple conditions jointly define what we're looking for, they MUST be preserved together in any subquestion. Failing to maintain these joint conditions completely invalidates the decomposition.
""".strip()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    
    return generate_response(messages, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)

#---------------------------------------------------- Answer Function -----------------------------------------------


#---------------------------------------------------- Get Final Answer Function -----------------------------------------------
def construct_final_prompt(current_question: str, current_sub_questions: List[Tuple[str, str]]) -> str:
    # Function to construct the final prompt remains unchanged
    examples = [
        {
            "question": "Which mountain is taller, Mount Everest or K2?",
            "sub_questions": [
                ("What is the height of Mount Everest?", "8,848 meters"),
                ("What is the height of K2?", "8,611 meters")
            ],
            "cot": "Mount Everest has a height of 8,848 meters, and K2 has a height of 8,611 meters. Since 8,848 meters is greater than 8,611 meters, Mount Everest is taller.",
            "final_answer": "Mount Everest"
        },
        {
            "question": "Who invented the telephone?",
            "sub_questions": [
                ("Which invention is known as the telephone?", "A device for voice communication"),
                ("Who is credited with creating the telephone?", "Alexander Graham Bell")
            ],
            "cot": "The telephone is a device for voice communication, and Alexander Graham Bell is credited with creating it.",
            "final_answer": "Alexander Graham Bell"
        },
          {
            "question": "What is the capital city of the country that hosts the annual Tour de France?",
            "sub_questions": [
                ("Which country hosts the annual Tour de France?", "France"),
                ("What is the capital city of France?", "Paris")
            ],
            "cot": "The country that hosts the annual Tour de France is France, and the capital city of France is Paris.",
            "final_answer": "Paris"
        },
        {
    "question": "How many players are on a standard soccer team during a match?",
    "sub_questions": [
        ("What is a standard soccer team?", "A team in an official soccer match"),
        ("How many players are on a standard soccer team?", "11")
    ],
    "cot": "A standard soccer team is a team in an official soccer match, and it has 11 players. Since the question asks 'how many players', the answer should include the unit 'players'.",
    "final_answer": "11 players"
},
        {
            "question": "Is the Pacific Ocean larger than the Atlantic Ocean?",
            "sub_questions": [
                ("What is the size of the Pacific Ocean?", "155.6 million square kilometers"),
                ("What is the size of the Atlantic Ocean?", "106.5 million square kilometers")
            ],
            "cot": "The Pacific Ocean has a size of 155.6 million square kilometers, and the Atlantic Ocean has a size of 106.5 million square kilometers. Since 155.6 million square kilometers is greater than 106.5 million square kilometers, the Pacific Ocean is larger.",
            "final_answer": "yes"
        },
        {
            "question": "When did the fictional character James Barker win his first Nobel Prize?",
            "sub_questions": [
                ("Who is James Barker?", "No reliable information found about a notable person named James Barker winning a Nobel Prize"),
                ("When did James Barker win his first Nobel Prize?", "No information available")
            ],
            "cot": "Based on the subquestions, there is no reliable information about a notable person named James Barker winning a Nobel Prize. The question appears to be asking about a fictional character or contains incorrect assumptions. Without concrete information about this person and their achievements, I cannot determine when they won their first Nobel Prize, or if they won one at all.",
            "final_answer": "[none]"
        },
        {
            "question": "Who was the first person to reach the summit of Mount Everest or K2?",
            "sub_questions": [
                ("Who was the first person to reach the summit of Mount Everest?", "Edmund Hillary and Tenzing Norgay in 1953"),
                ("Who was the first person to reach the summit of K2?", "[none]")
            ],
            "cot": "For Mount Everest, Edmund Hillary and Tenzing Norgay were the first to reach the summit in 1953. For K2, no information is available. Since the question asks for the first person to reach EITHER Mount Everest OR K2, and we have valid information for Mount Everest but not for K2, we can provide the answer based on the available information about Mount Everest.",
            "final_answer": "Edmund Hillary and Tenzing Norgay in 1953"
        }
    ]

    prompt = """You are an expert at answering questions based on provided subquestions and their answers. 

IMPORTANT GUIDELINES:
1. If you cannot extract the necessary information to answer the question, please respond with '[none]'.
2. Your answer MUST follow the structure "so the Final answer is: [your answer]".
3. First provide your chain of thought reasoning in the CoT section, then give your final answer.
Below are some examples to guide you:\n\n"""
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Question: {example['question']}\n"
        prompt += "Subquestions:\n"
        for sub_q, sub_a in example['sub_questions']:
            prompt += f"{sub_q}: {sub_a}\n"
        prompt += f"CoT: {example['cot']}\n"
        prompt += f"so the Final answer is: {example['final_answer']}\n\n"

    prompt += "Now, based on the following question and subquestions, generate the final answer:\n\n"
    prompt += f"Question: {current_question}\n"
    prompt += "Subquestions:\n"
    for sub_q, sub_a in current_sub_questions:
        prompt += f"{sub_q}: {sub_a}\n"
    prompt += """
CoT: 
"""
    return prompt

def get_final_answer(question: str, sub_questions: List[Tuple[str, str]], api_url=None) -> str:
    """Using OpenAI client to get the final answer"""
    prompt = construct_final_prompt(question, sub_questions)
    
    messages = [
        {"role": "system", "content": "You are an expert at answering questions based on provided subquestions and their answers."},
        {"role": "user", "content": prompt}
    ]
    
    return generate_response(messages, max_tokens=800, temperature=0)

#---------------------------------------------------- Parsing Response -----------------------------------------------
def parse_decomposition_response(response):
    # Function to parse decomposition response remains unchanged
    type_match = re.search(r'So the Type is:\s*(\w+)', response, re.IGNORECASE)
    question_type = type_match.group(1) if type_match else "None"
    
    # Extract Subquestion 1, ensuring only a single line of question content
    subq1_match = re.search(r'So the Subquestion 1 is:\s*(.+)', response, re.DOTALL)
    subq1 = subq1_match.group(1).strip() if subq1_match else ""
    # Clean up possible next line content
    subq1 = subq1.split('\n')[0].strip()  # Take only the first line
    
    subq2 = ""
    if question_type.lower() != "none":
        # Extract Subquestion 2, ensuring only a single line of question content
        subq2_match = re.search(r'So the Subquestion 2 is:\s*(.+)', response, re.DOTALL)
        subq2 = subq2_match.group(1).strip() if subq2_match else ""
        subq2 = subq2.split('\n')[0].strip()  # Take only the first line
        if not subq2 and question_type.lower() != "none":
            question_type = "None"
            
        # Check if subq2 contains [answer_subquestion1], if so ensure type is Sequential
        if "[answer_subquestion1]" in subq2:
            if question_type.lower() != "sequential":
                print(f"Detected subquestion2 containing reference to subquestion1 answer, correcting type from {question_type} to Sequential.")
                question_type = "Sequential"
        # If type is Sequential but subq2 doesn't reference subq1's answer, issue a warning but don't modify the type
        elif question_type.lower() == "sequential" and "[answer_subquestion1]" not in subq2:
            print(f"Warning: Type is Sequential but subquestion2 doesn't reference subquestion1's answer.")
    
    return {"type": question_type, "subq1": subq1, "subq2": subq2}

#---------------------------------------------------- Binary Tree Structure -----------------------------------------------
global_node_counter = 0
class QuestionNode:
    def __init__(self, question, q_type="None", subq1=None, subq2=None, parent=None, is_left_child=True, question_id=None):
        global global_node_counter
        # 如果没有提供ID，生成一个唯一ID
        if question_id is None:
            self.id = f"N{global_node_counter}"
            global_node_counter += 1
        else:
            self.id = question_id
        
        self.question = question
        self.display_question = question
        self.type = q_type
        self.left = None
        self.right = None
        self.parent = parent
        self.is_left_child = is_left_child
        self.depends_on = None
        self.subq1_text = subq1
        self.subq2_text = subq2
        self.answer = None  # Store the node's answer

    def __str__(self):
        return f"ID: {self.id}, Type: {self.type}, Q: {self.display_question[:50]}{'...' if len(self.display_question) > 50 else ''}"

#---------------------------------------------------- Extract Answer from Response -----------------------------------------------
def extract_answer(full_response):
    match = re.search(r'so the answer is:\s*(.*)', full_response, re.IGNORECASE)
    if match:
        raw_answer = match.group(1).strip()
        # Clean up the answer, remove extra symbols
        cleaned_answer = re.sub(r'["*]+$', '', raw_answer)
        # Handle [none] special case
        if cleaned_answer.lower() == '[none]':
            return '[none]'
        return cleaned_answer
    return "Answer not found"

#---------------------------------------------------- Tree Building and Traversal -----------------------------------------------
def build_question_tree(question, api_url=None, max_tokens=800, temperature=0.2, top_p=1.0, frequency_penalty=0.0, 
                        presence_penalty=0.0, examples_db=None, num_examples=3, depth=0, 
                        parent=None, is_left_child=True, max_height=3, placeholder_answers=None):
    """
    Build a question tree through recursive decomposition
    
    Parameters:
    - question: The question to decompose
    - api_url, max_tokens, temperature, top_p, frequency_penalty, presence_penalty: API parameters
    - examples_db: Database of examples for finding similar examples
    - num_examples: Number of similar examples to use
    - depth: Current depth in the tree (0 for root)
    - parent: Parent node
    - is_left_child: Whether the current node is a left child
    - max_height: Maximum allowed height of the tree (1 = root only)
    - placeholder_answers: Dictionary of previously computed answers
    
    Returns:
    - node: The root node of the built tree
    """
    global global_node_counter
    
 
    
    # Initialize placeholder_answers dictionary for storing replacement values
    if placeholder_answers is None:
        placeholder_answers = {}
    
    # Check if we've exceeded the maximum height
    # depth starts at 0 for root, so if max_height is 3, we allow depths 0, 1, and 2
    if depth >= max_height:
        
        # Create a leaf node with a unique ID using the global counter
        leaf_node = QuestionNode(
            question=question, 
            q_type="None",
            subq1=question, 
            subq2="",
            parent=parent, 
            is_left_child=is_left_child
        )
        
        
        
        # Handle replacement markers
        if parent and "[answer_subquestion1]" in leaf_node.question:
            if parent.type == "Sequential" and not is_left_child:
                leaf_node.depends_on = parent.left.id if parent.left else None
                leaf_node.display_question = leaf_node.question.replace("[answer_subquestion1]", f"[answer from {leaf_node.depends_on}]")

        elif parent and "[answer from" in leaf_node.question:
            match = re.search(r'\[answer from (.*?)\]', leaf_node.question)
            if match:
                leaf_node.depends_on = match.group(1)
                
        
   
        return leaf_node
    
    # Handle replacement markers, replace if replacement values are available
    modified_question = question
    depends_on = None
    
    if "[answer_subquestion1]" in question:
       
        if parent and parent.type == "Sequential" and not is_left_child and parent.left:
            depends_on = parent.left.id
           
            # Check if there's a replacement value available
            if depends_on in placeholder_answers:
                modified_question = question.replace("[answer_subquestion1]", placeholder_answers[depends_on])
                
    elif "[answer from" in question:
        
        match = re.search(r'\[answer from (.*?)\]', question)
        if match:
            depends_on = match.group(1)
           
            # Check if there's a replacement value available
            if depends_on in placeholder_answers:
                modified_question = question.replace(f"[answer from {depends_on}]", placeholder_answers[depends_on])
               
    
    # If the question still contains replacement markers and there's no replacement value available, create a node to be replaced
    if ("[answer_subquestion1]" in modified_question or "[answer from" in modified_question) and not "[answer from" in modified_question:
       
        node = QuestionNode(
            question=question, 
            q_type="None",
            subq1=question, 
            subq2="",
            parent=parent, 
            is_left_child=is_left_child
        )
        
        if depends_on:
            node.depends_on = depends_on
          
            if "[answer_subquestion1]" in node.question:
                node.display_question = node.question.replace("[answer_subquestion1]", f"[answer from {node.depends_on}]")
               

        return node
    
    # If examples_db is None, get the examples database
    if examples_db is None:
        examples_db = get_examples_database()
      
    
    # Analyze question structure and build decomposition prompt
  
    structure = analyze_question_structure(modified_question)
    print("++++++++++++++++++++++++")
    print(f"Question structure analysis (depth={depth}):", structure)
    similar_examples = find_similar_examples(modified_question, examples_db, num_examples)

    prompt = construct_prompt(modified_question, similar_examples, structure)

    response = generate_responses(prompt, api_url, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)
    print(f"Decomposition response (depth={depth}):", response)

    decomposition = parse_decomposition_response(response)
  

    # Create a new node with a unique ID
    node = QuestionNode(
        question=question, 
        q_type=decomposition['type'],
        subq1=decomposition['subq1'], 
        subq2=decomposition['subq2'],
        parent=parent, 
        is_left_child=is_left_child
    )
    
   
    if parent:
        print(f"Debugging - parent Node {parent.id}")
    
    # Use the modified question as the display question
    if modified_question != question:
        node.display_question = modified_question
        
    
    # Handle replacement markers
    if depends_on:
        node.depends_on = depends_on
       

    # Check if we need to stop expansion at this level to respect max_height
    # If we're at depth max_height-1, the next level would be at max_height
    # So we force this node to be a leaf by setting its type to "None"
    if depth >= max_height - 1:
        # Force node type to be "None" to prevent further decomposition
        node.type = "None"
        
        return node

    if node.type != "None":
      
        # First handle the left subtree
       
        left_node = build_question_tree(
            node.subq1_text, api_url, max_tokens, temperature, top_p,
            frequency_penalty, presence_penalty, examples_db, num_examples,
            depth + 1, node, True, max_height, placeholder_answers
        )
        node.left = left_node
        
        if node.left:
            print(f"Debugging - left Child node id: {node.left.id}, type: {node.left.type}")
        
        # Handle the right subtree
        if node.type == "Sequential":
            
            # For sequential type, check if the right subquestion contains replacement markers
            right_question = node.subq2_text
            
            right_node = build_question_tree(
                right_question, api_url, max_tokens, temperature, top_p,
                frequency_penalty, presence_penalty, examples_db, num_examples,
                depth + 1, node, False, max_height, placeholder_answers
            )
            node.right = right_node
            
            if node.right:
                print(f"Debugging - right Child node id: {node.right.id}, type: {node.right.type}")
            
            # Set up dependency relationships for sequential type
            if node.right and node.left:
                
                if node.right.depends_on is None and "[answer_subquestion1]" in node.right.question:
                    node.right.depends_on = node.left.id
                    node.right.display_question = node.right.question.replace("[answer_subquestion1]", f"[answer from {node.right.depends_on}]")
                    
        else:  # Parallel type
            
            right_node = build_question_tree(
                node.subq2_text, api_url, max_tokens, temperature, top_p,
                frequency_penalty, presence_penalty, examples_db, num_examples,
                depth + 1, node, False, max_height, placeholder_answers
            )
            node.right = right_node
            
            if node.right:
                print(f"Debugging - right Child node id: {node.right.id}, type: {node.right.type}")

    # Simple tree height and node count estimation
    has_children = node.left is not None or node.right is not None
    
    if has_children:
        print(f"Debugging - left child node: {' Exists' if node.left else 'does not exist '}, right child node: {' Exists' if node.right else' does not exist '}")
    
    
    return node



def get_all_nodes_postorder(node):
    """Get all nodes in post-order traversal"""
    if node is None:
        return []
    left_nodes = get_all_nodes_postorder(node.left)
    right_nodes = get_all_nodes_postorder(node.right)
    return left_nodes + right_nodes + [node]

def print_all_nodes(node, indent=""):
    # Function to print all nodes remains unchanged
    print(f"{indent}Node ID: {node.id}")
    print(f"{indent}Type: {node.type}")
    print(f"{indent}Question: {node.display_question}")
    if node.depends_on:
        print(f"{indent}Depends on: {node.depends_on}")
    if node.answer:
        print(f"{indent}Answer: {node.answer}")
    if node.left:
        print(f"{indent}Left child:")
        print_all_nodes(node.left, indent + "  ")
    if node.right:
        print(f"{indent}Right child:")
        print_all_nodes(node.right, indent + "  ")

#---------------------------------------------------- Tree Solving -----------------------------------------------
def solve_tree(root, original_question, api_url=None, max_tokens=4000, 
              temperature=0, top_p=1.0, frequency_penalty=0.0, 
              presence_penalty=0.0, examples_db=None, num_examples=20,
              enhanced_right_subtree=True, right_subtree_variants=2, 
              right_subtree_trees_per_variant=2, max_height=3,
              placeholder_answers=None):
    """
    Solve a single problem tree and return the answer, supporting the construction of an enhanced right subtree
    """
    global global_node_counter
    
    if placeholder_answers is None:
        placeholder_answers = {}  
    
    
    processed_node_ids = set()
    
    
    def solve_node(node, updated_tree=False, current_depth=0):
        if node is None:
            return {}

        
        if node.id in processed_node_ids:
            return {node.id: placeholder_answers.get(node.id, "[none]")}
        
        
        processed_node_ids.add(node.id)
        
        node_answers = {}
        
        
        if node.depends_on and node.depends_on not in placeholder_answers:
            
            def find_node_by_id(search_node, target_id):
                if search_node is None:
                    return None
                if search_node.id == target_id:
                    return search_node
                left_result = find_node_by_id(search_node.left, target_id)
                if left_result:
                    return left_result
                return find_node_by_id(search_node.right, target_id)
            
            dependent_node = find_node_by_id(root, node.depends_on)
            if dependent_node:
                dependent_answers = solve_node(dependent_node, updated_tree, current_depth)
                node_answers.update(dependent_answers)
        
       
        if node.id in placeholder_answers:
            node.answer = placeholder_answers[node.id]
            node_answers[node.id] = node.answer
            
            return node_answers
        
      
        if (node.left is None and node.right is None) or node.type == "None":
            actual_question = node.question
            
           
            if node.depends_on and node.depends_on in placeholder_answers:
                dependent_answer = placeholder_answers[node.depends_on]
                
                if dependent_answer.lower() == "[none]":
                    node.answer = "[none]"
                    placeholder_answers[node.id] = "[none]"
                    
                    node_answers[node.id] = "[none]"
                    return node_answers
                
                if "[answer_subquestion1]" in actual_question:
                    actual_question = actual_question.replace("[answer_subquestion1]", dependent_answer)
                elif f"[answer from {node.depends_on}]" in actual_question:
                    actual_question = actual_question.replace(f"[answer from {node.depends_on}]", dependent_answer)
                
                node.display_question = actual_question
            
            
            full_response = answer_question(
                question=actual_question,
                dataset=DATASET,
                method=METHOD,
                chunk_size=CHUNK_SIZE,
                min_sentence=MIN_SENTENCE,
                overlap=OVERLAP,
                topk1=TOPK1,
                topk2=TOPK2,
                max_iterations=MAX_ITERATIONS
            )
            answer = extract_answer(full_response)
            node.answer = answer
            placeholder_answers[node.id] = answer
            node_answers[node.id] = answer
            
            
            
            
            if (("[answer_subquestion1]" in node.question or "[answer from" in node.question) and 
                not updated_tree and node.answer and node.answer.lower() != "[none]"):
                
                
                if node.parent and not node.is_left_child:
                    print(f"node {node.id}: Contains the replacement tag and has obtained the answer. It needs to be rebuilt")
                    
            
            return node_answers
        
       
        left_answers = solve_node(node.left, updated_tree, current_depth + 1)
        node_answers.update(left_answers)
        
        
        needs_reconstruction = False
        if node.right and node.type == "Sequential":
            
            if ("[answer_subquestion1]" in node.right.question or 
                (node.right.depends_on and f"[answer from {node.right.depends_on}]" in node.right.question)):
                
               
                if node.left and node.left.id in placeholder_answers:
                    left_answer = placeholder_answers[node.left.id]
                   
                    if left_answer.lower() != "[none]":
                        needs_reconstruction = True
        
      
        if needs_reconstruction and not updated_tree and enhanced_right_subtree:
           
            
           
            new_right_question = generate_right_question_with_llm(
                parent_question=node.question,
                left_question=node.left.question if node.left else "",
                left_answer=placeholder_answers[node.left.id],
                original_right_question=node.right.question,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            
           
            
            
            remaining_height = max(max_height - current_depth - 1, 1) 
            
           
            new_right_node = build_enhanced_right_subtree(
                original_question=new_right_question,
                left_answer=placeholder_answers[node.left.id],
                api_url=api_url,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                examples_db=examples_db,
                num_examples=num_examples,
                max_height=remaining_height,  
                num_variants=right_subtree_variants,
                trees_per_variant=right_subtree_trees_per_variant
            )
            
            
            node.right = new_right_node
            
            
            
           
            right_answers = solve_node(node.right, True, current_depth + 1)
            node_answers.update(right_answers)
        elif needs_reconstruction and not updated_tree:
          
            
          
            new_right_question = generate_right_question_with_llm(
                parent_question=node.question,
                left_question=node.left.question if node.left else "",
                left_answer=placeholder_answers[node.left.id],
                original_right_question=node.right.question,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            
    
            
            
            remaining_height = max(max_height - current_depth - 1, 1)  
            
            
            new_right_node = build_question_tree(
                new_right_question, api_url, max_tokens, temperature, top_p,
                frequency_penalty, presence_penalty, examples_db, num_examples,
                depth=current_depth + 1, parent=node, is_left_child=False, 
                max_height=remaining_height, placeholder_answers=placeholder_answers
            )
            
          
            node.right = new_right_node
            
          

            right_answers = solve_node(node.right, True, current_depth + 1)
            node_answers.update(right_answers)
        else:
            
            if (node.right and ("[answer_subquestion1]" in node.right.question or 
                (node.right.depends_on and f"[answer from {node.right.depends_on}]" in node.right.question))):
                
               
                if node.left and node.left.id in placeholder_answers:
                    new_question = generate_right_question_with_llm(
                        parent_question=node.question,
                        left_question=node.left.question,
                        left_answer=placeholder_answers[node.left.id],
                        original_right_question=node.right.question
                    )
                    node.right.question = new_question
                    node.right.display_question = new_question
            
           
            right_answers = solve_node(node.right, updated_tree, current_depth + 1)
            node_answers.update(right_answers)
        
       
        child_questions = []
        valid_child_answers = False
        
       
        if node.left and node.left.id in node_answers:
            left_answer = node_answers[node.left.id]
            left_question = node.left.display_question
            child_questions.append((left_question, left_answer))
            
            if left_answer.lower() != "[none]":
                valid_child_answers = True
        
      
        if node.right and node.right.id in node_answers:
            right_answer = node_answers[node.right.id]
            right_question = node.right.display_question
            child_questions.append((right_question, right_answer))
           
            if right_answer.lower() != "[none]":
                valid_child_answers = True
        
       
        if node.type == "Sequential" and node.left and node.left.id in node_answers:
            if node_answers[node.left.id].lower() == "[none]":
                valid_child_answers = False
                
        
       
        if valid_child_answers and child_questions:
        
            final_answer = get_final_answer(node.display_question, child_questions, api_url)
            extracted_answer = re.search(r'final answer is:\s*(.*)', final_answer, re.DOTALL)
            if not extracted_answer:
                extracted_answer = re.search(r'answer is:\s*(.*)', final_answer, re.DOTALL)
            node.answer = extracted_answer.group(1).strip() if extracted_answer else final_answer
            placeholder_answers[node.id] = node.answer
            node_answers[node.id] = node.answer
            
            child_answers_str = ", ".join([f"「{q}」: 「{a}」" for q, a in child_questions])
           
            
            if node.answer.lower() == "[none]":
               
                full_response = answer_question(
                    node.display_question,
                    dataset=DATASET,
                    method=METHOD,
                    chunk_size=CHUNK_SIZE,
                    min_sentence=MIN_SENTENCE,
                    overlap=OVERLAP,
                    topk1=TOPK1,
                    topk2=TOPK2,
                    max_iterations=MAX_ITERATIONS
                )
                answer = extract_answer(full_response)
                node.answer = answer
                placeholder_answers[node.id] = answer
                node_answers[node.id] = answer
               
        else:

           
            full_response = answer_question(
                node.display_question,
                dataset=DATASET,
                method=METHOD,
                chunk_size=CHUNK_SIZE,
                min_sentence=MIN_SENTENCE,
                overlap=OVERLAP,
                topk1=TOPK1,
                topk2=TOPK2,
                max_iterations=MAX_ITERATIONS
            )
            answer = extract_answer(full_response)
            node.answer = answer
            placeholder_answers[node.id] = answer
            node_answers[node.id] = answer
           
        
        return node_answers
    
 
    all_answers = solve_node(root, False, 0)
    

    final_result = root.answer if root.answer else "[none]"
    
    return final_result


#---------------------------------------------------- Tree Statistics Collection -----------------------------------------------
def get_tree_statistics(root):
    """
    Calculate the statistics of a tree: height and node count
    Returns a tuple (height, node_count)
    """
    if root is None:
        return (0, 0)
    
    # Get all nodes
    all_nodes = get_all_nodes_postorder(root)
    node_count = len(all_nodes)
    
    # Calculate tree height
    def tree_height(node):
        if node is None:
            return 0
        return max(tree_height(node.left), tree_height(node.right)) + 1
    
    height = tree_height(root)
    
    return (height, node_count)



def serialize_question_tree(root):
    if root is None:
        return {"root_id": None, "nodes": []}

    nodes = []
    visited = set()

    def walk(node):
        if node is None or node.id in visited:
            return
        visited.add(node.id)
        nodes.append({
            "id": node.id,
            "type": node.type,
            "display_question": node.display_question,
            "answer": node.answer,
            "depends_on": node.depends_on,
            "left": node.left.id if node.left else None,
            "right": node.right.id if node.right else None,
        })
        walk(node.left)
        walk(node.right)

    walk(root)
    return {"root_id": root.id, "nodes": nodes}


def build_tree_shape_summary(tree_shape_counter):
    return [
        {
            "height": shape[0],
            "node_count": shape[1],
            "frequency": count,
        }
        for shape, count in tree_shape_counter.most_common()
    ]


#---------------------------------------------------- Multi-Tree Multi-Variant Solution -----------------------------------------------
def decompose_and_answer_with_variants(question, trees_per_question=TREES_PER_QUESTION, api_url=None, max_tokens=MAX_TOKENS, 
                                     temperature=DECOMPOSE_TEMPERATURE, top_p=TOP_P, frequency_penalty=FREQUENCY_PENALTY, presence_penalty=PRESENCE_PENALTY, 
                                     num_examples=NUM_EXAMPLES, max_height=MAX_HEIGHT, enhanced_right_subtree=ENHANCED_RIGHT_SUBTREE,
                                     right_subtree_variants=RIGHT_SUBTREE_VARIANTS, right_subtree_trees_per_variant=RIGHT_SUBTREE_TREES_PER_VARIANT,
                                     max_variants=MAX_VARIANTS, stats_file_path=STATS_FILE_PATH, question_started_at=None,
                                     question_deadline=None, timeout_budget_seconds=QUESTION_TIMEOUT_SECONDS, debug_collector=None):
    """
    Generate 6 trees per question, categorize by shape, and solve only one tree from the most common type.
    If unsuccessful, generate new question variants and try again.
    """
    global global_node_counter
    tree_decompose_started = time.perf_counter()

    try:
        # Reset global counter to ensure each call starts from 0
        global_node_counter = 0

        examples_db = get_examples_database()

        # Track attempt with original question and variants
        attempt_count = 0
        current_question = question
        attempted_questions = [question]

        # Variables to track tree heights
        initial_height = 0
        final_height = 0
        success = False

        def timed_out():
            return question_deadline is not None and time.perf_counter() >= question_deadline

        def timeout_elapsed_seconds():
            if question_started_at is None:
                return None
            return time.perf_counter() - question_started_at

        def finalize_timeout_fallback(stage, attempt_record=None):
            elapsed = timeout_elapsed_seconds()

            print(f"\n{'-'*80}")
            print(f"Question exceeded timeout budget at stage '{stage}'. Using direct lookup on original question.")
            print(f"{'-'*80}")

            if debug_collector is not None:
                debug_collector.set_timeout(
                    True,
                    stage=stage,
                    elapsed_seconds=elapsed,
                    budget_seconds=timeout_budget_seconds,
                )
                debug_collector.set_direct_fallback(True, reason=f"timeout:{stage}")
                debug_collector.add_error(
                    "tree_decompose.timeout",
                    f"Timeout budget exceeded at stage {stage}",
                    {
                        "stage": stage,
                        "timeout_budget_seconds": timeout_budget_seconds,
                        "timeout_elapsed_seconds": round(elapsed or 0.0, 4),
                    },
                )

            direct_fallback_started = time.perf_counter()
            final_answer = direct_answer(
                question,
                dataset=DATASET,
                method=METHOD,
                chunk_size=CHUNK_SIZE,
                min_sentence=MIN_SENTENCE,
                overlap=OVERLAP,
                topk1=TOPK1,
                topk2=TOPK2,
                debug_collector=debug_collector,
            )
            direct_fallback_elapsed = time.perf_counter() - direct_fallback_started

            if attempt_record is not None:
                attempt_record["final_answer"] = final_answer
                attempt_record["direct_fallback_seconds"] = direct_fallback_elapsed

            if debug_collector is not None:
                debug_collector.add_timing("direct_fallback_seconds", direct_fallback_elapsed)
                if attempt_record is not None:
                    if attempt_record.get("selected_tree"):
                        debug_collector.set_selected_tree(
                            {"attempt_index": attempt_record["attempt_index"], **attempt_record["selected_tree"]}
                        )
                    debug_collector.add_tree_attempt(attempt_record)

            save_tree_stats(question, final_answer, initial_height, final_height, stats_file_path, False)
            return final_answer

        while attempt_count <= max_variants:
            if timed_out():
                return finalize_timeout_fallback("before_attempt")
            attempt_started = time.perf_counter()
            attempt_record = {
                "attempt_index": attempt_count + 1,
                "question": current_question,
                "trees_generated": 0,
                "tree_shapes": [],
                "selected_tree": None,
                "final_answer": None,
                "solve_tree_seconds": 0.0,
                "variant_generation_seconds": 0.0,
                "attempt_total_seconds": 0.0,
            }

            print(f"\n{'='*80}")
            print(f"ATTEMPT {attempt_count + 1} with question: '{current_question}'")
            print(f"{'='*80}")

            all_trees = []

            print(f"\nGenerating {trees_per_question} trees for current question (max height: {max_height})")

            for j in range(trees_per_question):
                if timed_out():
                    attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                    return finalize_timeout_fallback("before_build_tree", attempt_record)

                tree_temp = 0
                print(f"\nBuilding tree {j + 1} for current question (temperature={tree_temp}, max_height={max_height}):")

                root = build_question_tree(
                    current_question,
                    api_url,
                    max_tokens,
                    tree_temp,
                    top_p,
                    frequency_penalty,
                    presence_penalty,
                    examples_db,
                    num_examples,
                    depth=0,
                    placeholder_answers={},
                    max_height=max_height,
                )

                height, node_count = get_tree_statistics(root)
                all_trees.append(
                    {
                        "tree": root,
                        "tree_num": j + 1,
                        "height": height,
                        "node_count": node_count,
                        "question_text": current_question,
                    }
                )

                print(f"Tree {j + 1} - Height: {height}, Node count: {node_count}")

            attempt_record["trees_generated"] = len(all_trees)

            tree_shape_counter = Counter()
            for tree_info in all_trees:
                shape = (tree_info["height"], tree_info["node_count"])
                tree_shape_counter[shape] += 1

            attempt_record["tree_shapes"] = build_tree_shape_summary(tree_shape_counter)

            print("\nTree shape frequencies (height, node count):")
            for shape, count in tree_shape_counter.most_common():
                print(f"Height: {shape[0]}, Node count: {shape[1]} - Frequency: {count}")

            if tree_shape_counter:
                most_common_shape, most_common_count = tree_shape_counter.most_common(1)[0]
                attempt_record["most_common_shape"] = {
                    "height": most_common_shape[0],
                    "node_count": most_common_shape[1],
                    "frequency": most_common_count,
                }
                print(
                    f"\nMost common shape: Height {most_common_shape[0]}, "
                    f"Node count {most_common_shape[1]} (Count: {most_common_count})"
                )

                most_common_trees = [
                    tree_info
                    for tree_info in all_trees
                    if (tree_info["height"], tree_info["node_count"]) == most_common_shape
                ]

                print(f"Found {len(most_common_trees)} trees with the most common shape")

                if most_common_trees:
                    tree_info = most_common_trees[0]
                    tree_root = tree_info["tree"]
                    question_text = tree_info["question_text"]
                    initial_height = tree_info["height"]

                    print(f"\n{'-'*80}")
                    print(f"Attempting to solve one tree from most common shape: Tree {tree_info['tree_num']}")
                    print(f"Question: '{question_text}'")
                    print(f"Tree height: {tree_info['height']}, Node count: {tree_info['node_count']}")
                    print(f"{'-'*80}")

                    print("\nTree structure:")
                    print_all_nodes(tree_root)

                    placeholder_answers = {}
                    if timed_out():
                        attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                        return finalize_timeout_fallback("before_solve_tree", attempt_record)
                    solve_tree_started = time.perf_counter()
                    try:
                        answer = solve_tree(
                            tree_root,
                            question_text,
                            api_url,
                            max_tokens,
                            temperature,
                            top_p,
                            frequency_penalty,
                            presence_penalty,
                            examples_db,
                            num_examples,
                            enhanced_right_subtree=enhanced_right_subtree,
                            right_subtree_variants=right_subtree_variants,
                            right_subtree_trees_per_variant=right_subtree_trees_per_variant,
                            max_height=max_height,
                            placeholder_answers=placeholder_answers,
                        )
                    except Exception as e:
                        attempt_record["solve_tree_seconds"] = time.perf_counter() - solve_tree_started
                        attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                        if debug_collector is not None:
                            debug_collector.add_error(
                                "tree_decompose.solve_tree",
                                str(e),
                                {
                                    "exception_type": type(e).__name__,
                                    "attempt_index": attempt_record["attempt_index"],
                                    "tree_num": tree_info["tree_num"],
                                    "solve_tree_seconds": round(attempt_record["solve_tree_seconds"], 4),
                                    "attempt_total_seconds": round(attempt_record["attempt_total_seconds"], 4),
                                },
                            )
                            debug_collector.add_tree_attempt(attempt_record)
                        raise

                    attempt_record["solve_tree_seconds"] = time.perf_counter() - solve_tree_started
                    final_height, _ = get_tree_statistics(tree_root)
                    attempt_record["selected_tree"] = {
                        "tree_num": tree_info["tree_num"],
                        "question": question_text,
                        "initial_height": initial_height,
                        "final_height": final_height,
                        "node_count": tree_info["node_count"],
                        "answer": answer,
                        "tree": serialize_question_tree(tree_root),
                    }
                    attempt_record["final_answer"] = answer

                    print(f"\nSelected tree returned answer: '{answer}'")
                    print(f"Initial tree height: {initial_height}, Final tree height after solving: {final_height}")

                    if answer.lower() != "[none]":
                        attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                        if debug_collector is not None:
                            debug_collector.set_selected_tree(
                                {"attempt_index": attempt_record["attempt_index"], **attempt_record["selected_tree"]}
                            )
                            debug_collector.add_tree_attempt(attempt_record)

                        print("Found valid answer, stopping tree traversal")
                        print("\n" + "=" * 80)
                        print(f"Final answer for question: '{question}'")
                        print(f"Answer: '{answer}'")
                        print("=" * 80)

                        success = True
                        save_tree_stats(question, answer, initial_height, final_height, stats_file_path, success)
                        return answer

                    if debug_collector is not None:
                        debug_collector.add_error(
                            "tree_decompose.none_answer",
                            "Selected tree returned [none]",
                            {
                                "attempt_index": attempt_record["attempt_index"],
                                "tree_num": tree_info["tree_num"],
                                "solve_tree_seconds": round(attempt_record["solve_tree_seconds"], 4),
                            },
                        )
                    print("Tree returned [none], will try with a new question variant")

            attempt_count += 1
            if attempt_count <= max_variants:
                print(f"\n{'-'*80}")
                print(f"No valid answer found from selected tree. Generating new question variant {attempt_count}")
                print(f"{'-'*80}")

                if timed_out():
                    attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                    return finalize_timeout_fallback("before_variant_generation", attempt_record)

                variant_started = time.perf_counter()
                new_variants = generate_question_variants(question, num_variants=1)
                attempt_record["variant_generation_seconds"] = time.perf_counter() - variant_started

                if len(new_variants) > 1:
                    new_question = new_variants[1]
                else:
                    print("Warning: generate_question_variants failed to produce a variant, falling back to original question")
                    if debug_collector is not None:
                        debug_collector.add_error(
                            "tree_decompose.variant_generation",
                            "Failed to produce a new variant, falling back to original question",
                            {
                                "attempt_index": attempt_count,
                                "variant_generation_seconds": round(attempt_record["variant_generation_seconds"], 4),
                            },
                        )
                    new_question = question

                print(f"New question variant {attempt_count}: '{new_question}'")
                attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                if debug_collector is not None:
                    if attempt_record.get("selected_tree"):
                        debug_collector.set_selected_tree(
                            {"attempt_index": attempt_record["attempt_index"], **attempt_record["selected_tree"]}
                        )
                    debug_collector.add_variant(
                        attempt_record["attempt_index"],
                        current_question,
                        new_question,
                        reason="generated_variant",
                        metadata={"duration_seconds": round(attempt_record["variant_generation_seconds"], 4)},
                    )
                    debug_collector.add_tree_attempt(attempt_record)
                current_question = new_question
                attempted_questions.append(current_question)
            else:
                print(f"\n{'-'*80}")
                print("Exhausted all variants. Using direct lookup on original question.")
                print(f"{'-'*80}")

                if debug_collector is not None:
                    debug_collector.set_direct_fallback(True, reason="exhausted_variants")
                    debug_collector.add_error(
                        "tree_decompose.direct_fallback",
                        "Exhausted variants and falling back to direct_answer",
                        {"attempt_count": attempt_count},
                    )

                direct_fallback_started = time.perf_counter()
                final_answer = direct_answer(
                    question,
                    dataset=DATASET,
                    method=METHOD,
                    chunk_size=CHUNK_SIZE,
                    min_sentence=MIN_SENTENCE,
                    overlap=OVERLAP,
                    topk1=TOPK1,
                    topk2=TOPK2,
                    debug_collector=debug_collector,
                )
                direct_fallback_elapsed = time.perf_counter() - direct_fallback_started

                attempt_record["attempt_total_seconds"] = time.perf_counter() - attempt_started
                attempt_record["final_answer"] = final_answer
                attempt_record["direct_fallback_seconds"] = direct_fallback_elapsed
                if debug_collector is not None:
                    debug_collector.add_timing("direct_fallback_seconds", direct_fallback_elapsed)
                    debug_collector.add_tree_attempt(attempt_record)

                success = False
                save_tree_stats(question, final_answer, initial_height, final_height, stats_file_path, success)
                return final_answer

        success = False
        save_tree_stats(question, "Could not determine an answer", initial_height, final_height, stats_file_path, success)
        return "Could not determine an answer after trying original question and variants."
    finally:
        if debug_collector is not None:
            debug_collector.add_timing("tree_decompose_total_seconds", time.perf_counter() - tree_decompose_started)



def main():
    """
    Main function that directly calls the question decomposition and answering functionality
    """
    # Set your question here or get user input at runtime
    question = """
Who is the son of the Italian navigator who explored the eastern coast of the continent Ulises Solís' birthplace is located in for England?
    """
    
    # Set maximum tree height
    max_height = 1
    
    # Set enhanced right subtree parameters
    enhanced_right_subtree = True
    right_subtree_variants = 2
    right_subtree_trees_per_variant = 3
    
    # Set file path for statistics
    stats_file_path = ""
    
    answer = decompose_and_answer_with_variants(
        question=question,
        trees_per_question=1,
        api_url=None,
        max_tokens=9000,
        temperature=0,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        num_examples=25,
        max_height=max_height,
        enhanced_right_subtree=enhanced_right_subtree,
        right_subtree_variants=right_subtree_variants, 
        right_subtree_trees_per_variant=right_subtree_trees_per_variant,  
        max_variants=0,
        stats_file_path=stats_file_path
    ) 
     
    # Print final result       
    print(f"\nFinal answer: {answer}")        
 
# Run the main function directly 
if __name__ == "__main__":        
    main()   