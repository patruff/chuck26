from smolagents import Tool

class WordPhoneTool(Tool):
    name = "word_phonetic_analyzer" 
    description = """Analyzes word pronunciation using CMU dictionary and custom pronunciations to get phonemes, syllables, and stress patterns. 
    Can also compare two words for phonetic similarity and rhyming."""
    
    inputs = {
        "word": {
            "type": "string",
            "description": "Primary word to analyze for pronunciation patterns"
        },
        "compare_to": {
            "type": "string",
            "description": "Optional word to compare against for similarity scoring",
            "nullable": True
        },
        "custom_phones": {
            "type": "object",
            "description": "Optional dictionary of custom word pronunciations",
            "nullable": True
        }
    }
    output_type = "string"

    VOWEL_REF = "AH,UH,AX|AE,EH|IY,IH|AO,AA|UW,UH|AY,EY|OW,AO|AW,AO|OY,OW|ER,AXR"

    def _get_vowel_groups(self):
        groups = []
        group_strs = self.VOWEL_REF.split("|")
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

    def _get_last_syllable(self, phones):
        last_vowel_idx = -1
        last_vowel = None
        vowel_groups = self._get_vowel_groups()
        
        for i in range(len(phones)):
            phone = phones[i]
            base_phone = ""
            for j in range(len(phone)):
                char = phone[j]
                if char not in "012":
                    base_phone += char
            
            for group in vowel_groups:
                if base_phone in group:
                    last_vowel_idx = i
                    last_vowel = base_phone
                    break
        
        if last_vowel_idx == -1:
            return None, []
            
        remaining = []
        for i in range(last_vowel_idx + 1, len(phones)):
            remaining.append(phones[i])
            
        return last_vowel, remaining

    def _strip_stress(self, phones):
        result = []
        for phone in phones:
            stripped = ""
            for char in phone:
                if char not in "012":
                    stripped += char
            result.append(stripped)
        return result

    def _vowels_match(self, v1, v2):
        v1_stripped = ""
        v2_stripped = ""
        
        for char in v1:
            if char not in "012":
                v1_stripped += char
                
        for char in v2:
            if char not in "012":
                v2_stripped += char
        
        if v1_stripped == v2_stripped:
            return True
            
        vowel_groups = self._get_vowel_groups()
        for group in vowel_groups:
            if v1_stripped in group and v2_stripped in group:
                return True
        return False

    def _calculate_similarity(self, word1, phones1, word2, phones2):
        import pronouncing
        
        # Initialize variables before use
        last_vowel1 = None
        last_vowel2 = None
        word1_end = []
        word2_end = []
        matched = 0
        common_length = 0
        end1_clean = []
        end2_clean = []
        i = 0  # Initialize i for loop variable
        
        phone_list1 = phones1.split()
        phone_list2 = phones2.split()
        
        # Get last syllable components
        result1 = self._get_last_syllable(phone_list1)
        result2 = self._get_last_syllable(phone_list2)
        last_vowel1, word1_end = result1
        last_vowel2, word2_end = result2
        
        # Calculate length similarity score first
        phone_diff = abs(len(phone_list1) - len(phone_list2))
        max_phones = max(len(phone_list1), len(phone_list2))
        length_score = 1.0 if phone_diff == 0 else 1.0 - (phone_diff / max_phones)
        
        # Calculate rhyme score (most important)
        rhyme_score = 0.0
        if last_vowel1 and last_vowel2:
            if self._vowels_match(last_vowel1, last_vowel2):
                end1_clean = self._strip_stress(word1_end)
                end2_clean = self._strip_stress(word2_end)
                
                if end1_clean == end2_clean:
                    rhyme_score = 1.0  # Perfect rhyme, capped at 1.0
                else:
                    # Partial rhyme based on ending similarity
                    common_length = min(len(end1_clean), len(end2_clean))
                    matched = 0
                    for i in range(common_length):
                        if end1_clean[i] == end2_clean[i]:
                            matched += 1
                    rhyme_score = 0.6 * (matched / max(len(end1_clean), len(end2_clean)))

        # Calculate stress pattern similarity
        stress1 = pronouncing.stresses(phones1)
        stress2 = pronouncing.stresses(phones2)
        stress_score = 1.0 if stress1 == stress2 else 0.5
        
        # Weighted combination prioritizing rhyming and length
        total_similarity = (
            (rhyme_score * 0.6) +       # Rhyming most important (60%)
            (length_score * 0.3) +      # Length similarity next (30%)
            (stress_score * 0.1)        # Stress pattern least important (10%)
        )
        
        # Ensure total similarity is capped at 1.0
        total_similarity = min(1.0, total_similarity)
        
        return {
            "similarity": round(total_similarity, 3),
            "rhyme_score": round(rhyme_score, 3),
            "length_score": round(length_score, 3),
            "stress_score": round(stress_score, 3),
            "phone_length_difference": phone_diff
        }

    def forward(self, word, compare_to=None, custom_phones=None):
        import json
        import string
        import pronouncing
        
        # Initialize variables before use
        word_last_vowel = None
        compare_last_vowel = None
        word_end = []
        compare_end = []
        is_rhyme = False
        
        word_clean = word.lower()
        word_clean = word_clean.strip(string.punctuation)
        primary_phones = self._get_word_phones(word_clean, custom_phones)
        
        if not primary_phones:
            result = {
                'word': word_clean, 
                'found': False,
                'error': 'Word not found in dictionary or custom phones'
            }
            return json.dumps(result, indent=2)
        
        result = {
            'word': word_clean,
            'found': True,
            'analysis': {
                'syllable_count': pronouncing.syllable_count(primary_phones),
                'phones': primary_phones.split(),
                'stresses': pronouncing.stresses(primary_phones),
                'phone_count': len(primary_phones.split())
            }
        }
        
        if compare_to:
            compare_clean = compare_to.lower()
            compare_clean = compare_clean.strip(string.punctuation)
            compare_phones = self._get_word_phones(compare_clean, custom_phones)
            
            if not compare_phones:
                result['comparison'] = {
                    'error': f'Comparison word "{compare_clean}" not found in dictionary or custom phones'
                }
            else:
                # Get rhyme components
                word_result = self._get_last_syllable(primary_phones.split())
                compare_result = self._get_last_syllable(compare_phones.split())
                word_last_vowel, word_end = word_result
                compare_last_vowel, compare_end = compare_result
                
                # Calculate if words rhyme
                if word_last_vowel and compare_last_vowel:
                    if self._vowels_match(word_last_vowel, compare_last_vowel):
                        word_end_clean = self._strip_stress(word_end)
                        compare_end_clean = self._strip_stress(compare_end)
                        if word_end_clean == compare_end_clean:
                            is_rhyme = True
                
                # Calculate detailed comparison stats
                word_syl_count = pronouncing.syllable_count(primary_phones)
                compare_syl_count = pronouncing.syllable_count(compare_phones)
                
                result['comparison'] = {
                    'word': compare_clean,
                    'analysis': {
                        'syllable_count': compare_syl_count,
                        'phones': compare_phones.split(),
                        'stresses': pronouncing.stresses(compare_phones),
                        'phone_count': len(compare_phones.split())
                    },
                    'comparison_stats': {
                        'is_rhyme': is_rhyme,
                        'same_syllable_count': word_syl_count == compare_syl_count,
                        'same_stress_pattern': pronouncing.stresses(primary_phones) == pronouncing.stresses(compare_phones),
                        'syllable_difference': abs(word_syl_count - compare_syl_count),
                        'phone_difference': abs(len(primary_phones.split()) - len(compare_phones.split()))
                    }
                }
                
                # Calculate detailed similarity scores
                similarity_result = self._calculate_similarity(
                    word_clean, primary_phones,
                    compare_clean, compare_phones
                )
                result['similarity'] = similarity_result
        
        return json.dumps(result, indent=2)
