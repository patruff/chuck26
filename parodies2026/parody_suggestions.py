from smolagents import Tool

class ParodyWordSuggestionTool(Tool):
    name = "parody_word_suggester"
    description = """Suggests rhyming funny words using CMU dictionary and custom pronunciations.
    Returns similar-sounding words that rhyme, especially focusing on common vowel sounds."""
    
    inputs = {
        "target": {
            "type": "string",
            "description": "The word you want to find rhyming alternatives for"
        },
        "word_list_str": {
            "type": "string",
            "description": "JSON string of word list (e.g. '[\"word1\", \"word2\"]')"
        },
        "min_similarity": {
            "type": "string",
            "description": "Minimum similarity threshold (0.0-1.0)",
            "nullable": True,
            "default": "0.6"
        },
        "custom_phones": {
            "type": "object",
            "description": "Optional dictionary of custom word pronunciations",
            "nullable": True,
            "default": None
        }
    }
    output_type = "string"

    # Vowel reference groups
    VOWEL_REF = "AH,UH,AX|AE,EH|IY,IH|AO,AA|UW,UH|AY,EY|OW,AO|AW,AO|OY,OW|ER,AXR"

    def _get_vowel_groups(self):
        groups = []
        group_strs = self.VOWEL_REF.split("|")
        group_str = ""
        for group_str in group_strs:
            groups.append(group_str.split(","))
        return groups

    def _get_word_phones(self, word, custom_phones=None):
        """Get phones for a word, checking custom dictionary first."""
        if custom_phones and word in custom_phones:
            return custom_phones[word]["primary_phones"]
            
        import pronouncing
        phones = pronouncing.phones_for_word(word)
        return phones[0] if phones else None

    def _get_last_syllable(self, phones: list) -> tuple:
        """Extract the last syllable (vowel + remaining consonants)."""
        last_vowel_idx = -1
        last_vowel = None
        vowel_groups = self._get_vowel_groups()
        
        # Initialize loop variables
        i = 0
        phone = ""
        base_phone = ""
        group = []
        vowel_char = ""
        
        # First, find the primary stressed vowel if it exists
        for i, phone in enumerate(phones):
            # Check for primary stress (1)
            if '1' in phone:
                # Check if it's a vowel
                base_phone = phone.rstrip('012')
                for vowel_char in 'AEIOU':
                    if vowel_char in base_phone:
                        last_vowel_idx = i
                        last_vowel = base_phone
                        break
                if last_vowel is not None:
                    break
        
        # If no primary stress, just use the last vowel
        if last_vowel_idx == -1:
            for i, phone in enumerate(phones):
                base_phone = phone.rstrip('012')
                for vowel_char in 'AEIOU':
                    if vowel_char in base_phone:
                        last_vowel_idx = i
                        last_vowel = base_phone
        
        if last_vowel_idx == -1:
            return None, []
            
        remaining = phones[last_vowel_idx + 1:]
        return last_vowel, remaining

    def _strip_stress(self, phones: list) -> list:
        """Remove stress markers from phones."""
        result = []
        # Initialize loop variable
        phone = ""
        
        for phone in phones:
            result.append(phone.rstrip('012'))
        return result

    def _vowels_match(self, v1: str, v2: str) -> bool:
        """Check if vowels belong to the same sound group."""
        v1 = v1.rstrip('012')
        v2 = v2.rstrip('012')
        
        if v1 == v2:
            return True
        
        # Initialize loop variables
        vowel_groups = self._get_vowel_groups()
        group = []
        
        for group in vowel_groups:
            if v1 in group and v2 in group:
                return True
        return False
    
    def _consonants_are_similar(self, c1, c2):
        """Check if two consonants belong to similar phonetic groups."""
        # Group consonants by articulation manner
        nasals = ['M', 'N', 'NG']
        stops = ['P', 'B', 'T', 'D', 'K', 'G']
        fricatives = ['F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH']
        liquids = ['L', 'R']
        glides = ['W', 'Y']
        
        # Check if consonants are in the same group
        if c1 in nasals and c2 in nasals:
            return True
        if c1 in stops and c2 in stops:
            return True
        if c1 in fricatives and c2 in fricatives:
            return True
        if c1 in liquids and c2 in liquids:
            return True
        if c1 in glides and c2 in glides:
            return True
        
        return False
    
    def _words_have_similar_structure(self, word1, word2, phones1, phones2):
        """Check if words have similar structure beyond just ending."""
        # Similar word length
        if abs(len(word1) - len(word2)) > 2:
            return False
            
        # Similar syllable count
        import pronouncing
        syllables1 = len(pronouncing.stresses(phones1))
        syllables2 = len(pronouncing.stresses(phones2))
        if syllables1 != syllables2:
            return False
            
        # For -ing words, check if consonants before -ing have similar patterns
        if word1.endswith('ing') and word2.endswith('ing'):
            # Get consonant patterns (c-v-c structure)
            phone_list1 = phones1.split()
            phone_list2 = phones2.split()
            
            # Initialize variables for list comprehension
            p = ""
            v = ""
            
            # Get consonants
            consonants1 = [p for p in self._strip_stress(phone_list1) if not any(v in p for v in 'AEIOU')]
            consonants2 = [p for p in self._strip_stress(phone_list2) if not any(v in p for v in 'AEIOU')]
            
            # Same consonant count is promising
            if len(consonants1) == len(consonants2):
                return True
                
            # For words like 'running' and 'cumming', check pre-final consonant similarity
            if len(consonants1) >= 2 and len(consonants2) >= 2:
                pre_final1 = consonants1[-2]
                pre_final2 = consonants2[-2]
                if pre_final1 == pre_final2 or self._consonants_are_similar(pre_final1, pre_final2):
                    return True
                    
        return False

    def _calculate_similarity(self, word1, phones1, word2, phones2):
        """Calculate similarity score using refined metrics for parody."""
        # Initialize all variables
        phone_list1 = phones1.split()
        phone_list2 = phones2.split()
        
        # Variables for rhyme scoring
        rhyme_score = 0.0
        word_vowel = None
        word_end = []
        target_vowel = None
        target_end = []
        word_end_clean = []
        target_end_clean = []
        common_length = 0
        matched = 0
        i = 0
        
        # Variables for whole-word matching
        primary_stress_vowel1 = None
        primary_stress_vowel2 = None
        primary_stress_idx1 = -1
        primary_stress_idx2 = -1
        front_consonants1 = []
        front_consonants2 = []
        
        # Variables for special pattern matching
        special_pattern_score = 0.0
        stem1 = ""
        stem2 = ""
        consonant1 = ""
        consonant2 = ""
        nasals = ['m', 'n']
        stops = ['p', 'b', 't', 'd', 'k', 'g']
        fricatives = ['f', 'v', 'th', 's', 'z', 'sh']
        base1 = ""
        base2 = ""
        
        # Variables for list comprehensions
        p = ""
        v = ""
        group = []
        
        # Find primary stressed vowels
        for i, phone in enumerate(phone_list1):
            if '1' in phone and any(v in phone for v in 'AEIOU'):
                primary_stress_vowel1 = phone.rstrip('012')
                primary_stress_idx1 = i
                break
                
        for i, phone in enumerate(phone_list2):
            if '1' in phone and any(v in phone for v in 'AEIOU'):
                primary_stress_vowel2 = phone.rstrip('012')
                primary_stress_idx2 = i
                break
        
        # Get consonants before the primary stress
        if primary_stress_idx1 > 0:
            front_consonants1 = [p for p in self._strip_stress(phone_list1[:primary_stress_idx1]) 
                                if not any(v in p for v in 'AEIOU')]
                                
        if primary_stress_idx2 > 0:
            front_consonants2 = [p for p in self._strip_stress(phone_list2[:primary_stress_idx2]) 
                                if not any(v in p for v in 'AEIOU')]
        
        # Calculate front consonant similarity (important for parody)
        front_consonant_score = 0.0
        if front_consonants1 and front_consonants2:
            min_length = min(len(front_consonants1), len(front_consonants2))
            if min_length > 0:
                matches = 0
                for i in range(min_length):
                    if front_consonants1[i] == front_consonants2[i]:
                        matches += 1
                front_consonant_score = matches / min_length
        
        # Get last syllable components for rhyming
        result1 = self._get_last_syllable(phone_list1)
        result2 = self._get_last_syllable(phone_list2)
        word_vowel, word_end = result1
        target_vowel, target_end = result2
        
        # Perfect rhyme check (40% of score)
        if word_vowel and target_vowel:
            if self._vowels_match(word_vowel, target_vowel):
                word_end_clean = self._strip_stress(word_end)
                target_end_clean = self._strip_stress(target_end)
                
                if word_end_clean == target_end_clean:
                    rhyme_score = 1.0
                else:
                    # Partial rhyme based on ending similarity
                    common_length = min(len(word_end_clean), len(target_end_clean))
                    matched = 0
                    for i in range(common_length):
                        if word_end_clean[i] == target_end_clean[i]:
                            matched += 1
                    if max(len(word_end_clean), len(target_end_clean)) > 0:
                        rhyme_score = 0.6 * (matched / max(1, max(len(word_end_clean), len(target_end_clean))))
                    else:
                        rhyme_score = 0.6  # Still somewhat rhymes even without ending consonants
        
        # Primary stressed vowel match (15% of score)
        primary_vowel_score = 0.0
        if primary_stress_vowel1 and primary_stress_vowel2:
            if primary_stress_vowel1 == primary_stress_vowel2:
                primary_vowel_score = 1.0
            else:
                # Check if they're in the same vowel group
                for group in self._get_vowel_groups():
                    if primary_stress_vowel1 in group and primary_stress_vowel2 in group:
                        primary_vowel_score = 0.7
                        break
        
        # Near rhyme check - 15% of score
        near_rhyme_score = 0.0
        
        # Check for specific endings
        if len(phone_list1) >= 2 and len(phone_list2) >= 2:
            # Check for -ing endings
            if (self._strip_stress(phone_list1[-2:]) == ['IH', 'NG'] and 
                self._strip_stress(phone_list2[-2:]) == ['IH', 'NG']):
                
                # Base score for -ing endings
                near_rhyme_score = 0.6
                
                # Additional checks for consonant before -ing
                if len(phone_list1) >= 3 and len(phone_list2) >= 3:
                    consonant1_list = self._strip_stress(phone_list1[-3:-2])
                    consonant2_list = self._strip_stress(phone_list2[-3:-2])
                    
                    if consonant1_list and consonant2_list:
                        consonant1 = consonant1_list[0]
                        consonant2 = consonant2_list[0]
                        
                        # Same consonant gets highest score (like running/gunning)
                        if consonant1 == consonant2:
                            near_rhyme_score = 0.9
                        # Similar consonants (nasal: 'N'/'M') get high score (running/cumming)
                        elif self._consonants_are_similar(consonant1, consonant2):
                            near_rhyme_score = 0.8
                            
            # Check for -y endings (like happy/sappy) 
            elif (self._strip_stress(phone_list1[-1:]) == ['IY'] and 
                  self._strip_stress(phone_list2[-1:]) == ['IY']):
                near_rhyme_score = 0.7
        
        # Special pattern matching for running/cumming type pairs (15% of score)
        if word1.endswith('ing') and word2.endswith('ing'):
            # Get the stem (without -ing)
            stem1 = word1[:-3]
            stem2 = word2[:-3]
            
            # Same stem length is good for parody
            if len(stem1) == len(stem2):
                special_pattern_score += 0.4
                
            # If both stems end with same consonant (like 'n' in run-ning, 'm' in cum-ming)
            # this makes them rhyme much better
            if stem1 and stem2 and stem1[-1] == stem2[-1]:
                special_pattern_score += 0.3
            elif stem1 and stem2:
                # Check if the final consonants are in the same phonetic group
                # This helps pair words like running/humming (nasal consonants)
                consonant1 = stem1[-1]
                consonant2 = stem2[-1]
                
                # Check if they're in the same group
                if (consonant1 in nasals and consonant2 in nasals) or \
                   (consonant1 in stops and consonant2 in stops) or \
                   (consonant1 in fricatives and consonant2 in fricatives):
                    special_pattern_score += 0.2
                    
            # Check for double consonants (like nn in running, mm in cumming)
            if len(stem1) >= 2 and stem1[-1] == stem1[-2] and \
               len(stem2) >= 2 and stem2[-1] == stem2[-2]:
                special_pattern_score += 0.3
        
        # Length and stress similarity (5% each)
        phone_diff = abs(len(phone_list1) - len(phone_list2))
        max_phones = max(len(phone_list1), len(phone_list2))
        length_score = 1.0 if phone_diff == 0 else 1.0 - (phone_diff / max_phones)
        
        # Check stress pattern similarity
        import pronouncing
        stress1 = pronouncing.stresses(phones1)
        stress2 = pronouncing.stresses(phones2)
        stress_score = 1.0 if stress1 == stress2 else 0.5
        
        # Front consonant match (5% of score)
        front_score = front_consonant_score * 0.05
        
        # Weighted combination
        similarity = (
            (rhyme_score * 0.40) +           # End rhyme (40%)
            (primary_vowel_score * 0.15) +   # Primary vowel (15%)
            (near_rhyme_score * 0.15) +      # Near rhyme features (15%)
            (special_pattern_score * 0.15) + # Special pattern match (15%)
            (length_score * 0.05) +          # Length similarity (5%)
            (stress_score * 0.05) +          # Stress pattern (5%)
            (front_score)                    # Front consonants (5%)
        )
        
        # Additional boost for specific word patterns that make great parody matches
        # This specifically addresses running/cumming type pairs
        if word1.endswith('ing') and word2.endswith('ing'):
            base1 = word1[:-3]
            base2 = word2[:-3]
            
            # Specific pattern for words like running/cunning/cumming
            if (len(base1) == 3 and len(base2) == 3 and
                base1[0] != base2[0] and     # Different first consonant (good for parody)
                len(base1) >= 2 and len(base2) >= 2 and
                base1[-1] == base1[-2] and   # Double consonant in first word (nn in running)
                base2[-1] == base2[-2]):     # Double consonant in second word (mm in cumming)
                similarity = max(similarity, 0.9)  # These are excellent parody matches
        
        # Cap at 1.0
        similarity = min(1.0, similarity)
        
        return {
            "similarity": round(similarity, 3),
            "rhyme_score": round(rhyme_score, 3),
            "primary_vowel_score": round(primary_vowel_score, 3),
            "near_rhyme_score": round(near_rhyme_score, 3),
            "special_pattern_score": round(special_pattern_score, 3),
            "length_score": round(length_score, 3),
            "stress_score": round(stress_score, 3),
            "front_consonant_score": round(front_consonant_score, 3),
            "phone_length_difference": phone_diff
        }

    def forward(self, target: str, word_list_str: str, min_similarity: str = "0.6", custom_phones: dict = None) -> str:
        import pronouncing
        import string
        import json
        
        # Initialize all variables
        target = target.lower().strip(string.punctuation)
        min_similarity = float(min_similarity)
        suggestions = []
        valid_words = []
        invalid_words = []
        words = []
        target_phones = ""
        target_phone_list = []
        target_vowel = None
        target_end = []
        word = ""
        word_phones = ""
        word_phone_list = []
        word_vowel = None
        word_end = []
        similarity_result = {}
        
        # Parse JSON string to list
        try:
            words = json.loads(word_list_str)
        except json.JSONDecodeError:
            return json.dumps({
                "error": "Invalid JSON string for word_list_str",
                "suggestions": []
            }, indent=2)
            
        # Get target pronunciation
        target_phones = self._get_word_phones(target, custom_phones)
        if not target_phones:
            return json.dumps({
                "error": f"Target word '{target}' not found in dictionary or custom phones",
                "suggestions": []
            }, indent=2)
        
        # Parse target phones
        target_phone_list = target_phones.split()
        target_vowel, target_end = self._get_last_syllable(target_phone_list)
        
        # Filter word list
        for word in words:
            word = word.lower().strip(string.punctuation)
            if self._get_word_phones(word, custom_phones):
                valid_words.append(word)
            else:
                invalid_words.append(word)
        
        if not valid_words:
            return json.dumps({
                "error": "No valid words found in dictionary or custom phones",
                "invalid_words": invalid_words,
                "suggestions": []
            }, indent=2)
        
        # Check each word
        for word in valid_words:
            word_phones = self._get_word_phones(word, custom_phones)
            if word_phones:
                similarity_result = self._calculate_similarity(word, word_phones, target, target_phones)
                
                if similarity_result["similarity"] >= min_similarity:
                    word_phone_list = word_phones.split()
                    word_vowel, word_end = self._get_last_syllable(word_phone_list)
                    
                    suggestions.append({
                        "word": word,
                        "similarity": similarity_result["similarity"],
                        "rhyme_score": similarity_result["rhyme_score"],
                        "primary_vowel_score": similarity_result["primary_vowel_score"],
                        "near_rhyme_score": similarity_result["near_rhyme_score"],
                        "special_pattern_score": similarity_result.get("special_pattern_score", 0),
                        "length_score": similarity_result["length_score"],
                        "stress_score": similarity_result["stress_score"],
                        "front_consonant_score": similarity_result["front_consonant_score"],
                        "phones": word_phones,
                        "last_vowel": word_vowel,
                        "ending": " ".join(word_end) if word_end else "",
                        "is_custom": word in custom_phones if custom_phones else False
                    })
        
        # Sort by similarity score descending
        suggestions.sort(key=lambda x: x["similarity"], reverse=True)
        
        result = {
            "target": target,
            "target_phones": target_phones,
            "target_last_vowel": target_vowel,
            "target_ending": " ".join(target_end) if target_end else "",
            "invalid_words": invalid_words,
            "suggestions": suggestions
        }
        
        return json.dumps(result, indent=2)
