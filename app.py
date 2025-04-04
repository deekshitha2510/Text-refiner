from flask import Flask, render_template, request, jsonify
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import wordnet
from nltk import pos_tag
from collections import defaultdict
import re
from textblob import TextBlob
from textblob import Word
from flask_cors import CORS 

app = Flask(__name__)
CORS(app) 

class GrammarChecker:
    def __init__(self):
        self._download_nltk_resources()
        self.grammar_rules = {
            'NN': 'Noun',
            'NNS': 'Noun (plural)',
            'VB': 'Verb',
            'JJ': 'Adjective',
            'RB': 'Adverb',
            'PRP': 'Pronoun',
            'DT': 'Determiner',
            'IN': 'Preposition',
            'CC': 'Conjunction'
        }
    
    def _download_nltk_resources(self):
        resources = [
            ('averaged_perceptron_tagger', 'taggers'),
            ('punkt', 'tokenizers'),
            ('wordnet', 'corpora'),
            ('omw-1.4', 'corpora'),
            ('words', 'corpora')
        ]
        for resource, package in resources:
            try:
                nltk.data.find(f'{package}/{resource}')
            except LookupError:
                nltk.download(resource)
    
    def mark_grammar_errors(self, text):
        sentences = sent_tokenize(text)
        results = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            tagged = pos_tag(tokens)
            
            analysis = []
            for word, tag in tagged:
                pos = self.grammar_rules.get(tag[:2], 'Other')
                spelling_issues = self._check_spelling(word)
                analysis.append({
                    'word': word,
                    'pos': pos,
                    'tag': tag,
                    'spelling_issues': spelling_issues
                })
            
            results.append({
                'sentence': sentence,
                'analysis': analysis
            })
        
        return {
            'original_text': text,
            'analysis': results,
            'summary': self._generate_summary(results)
        }
    
    def _check_spelling(self, word):
        # Skip non-alphabetic words or very short words
        if not word.isalpha() or len(word) <= 1:
            return None
            
        # Skip proper nouns (capitalized words)
        if word[0].isupper():
            return None
            
        # Check against NLTK words corpus
        try:
            from nltk.corpus import words as nltk_words
            if word.lower() in nltk_words.words():
                return None
        except:
            pass
            
        # Check against WordNet
        if wordnet.synsets(word.lower()):
            return None
            
        # Use TextBlob for spelling correction
        word_obj = Word(word.lower())
        suggestions = word_obj.spellcheck()
        
        if suggestions and suggestions[0][0].lower() != word.lower():
            # If the top suggestion is different and confidence is reasonable
            if suggestions[0][1] > 0.7:  # Increased confidence threshold
                return {
                    'incorrect': True,
                    'suggestions': [s[0] for s in suggestions[:3] if s[0].lower() != word.lower()]
                }
        return None
    
    def _generate_summary(self, analysis):
        pos_counts = defaultdict(int)
        spelling_errors = 0
        for sentence in analysis:
            for word in sentence['analysis']:
                pos_counts[word['pos']] += 1
                if word['spelling_issues'] and word['spelling_issues']['incorrect']:
                    spelling_errors += 1
        
        summary = dict(pos_counts)
        summary['spelling_errors'] = spelling_errors
        return summary

class PlainEnglishParaphraser:
    def __init__(self):
        self._download_nltk_resources()
        self.phrase_replacements = {
            'in spite of the fact that': 'although',
            'at this point in time': 'now',
            'the majority of': 'most',
            'are in agreement': 'agree',
            'not too distant future': 'soon',
            'may be eliminated altogether': 'may be cut',
            'there is a solemn danger': 'there is a risk',
            'professional nurses': 'school nurses',
            'are essential': 'are needed'
        }
        self.word_replacements = {
            'solemn': 'serious',
            'essential': 'necessary',
            'roles': 'positions',
            'funding': 'money',
            'individuals': 'people',
            'utilize': 'use',
            'commence': 'start',
            'terminate': 'end',
            'approximately': 'about',
            'facilitate': 'help'
        }
    
    def _download_nltk_resources(self):
        resources = [
            ('averaged_perceptron_tagger', 'taggers'),
            ('punkt', 'tokenizers'),
            ('wordnet', 'corpora'),
            ('omw-1.4', 'corpora'),
            ('words', 'corpora')
        ]
        for resource, package in resources:
            try:
                nltk.data.find(f'{package}/{resource}')
            except LookupError:
                nltk.download(resource)
    
    def _simplify_phrase(self, text):
        for phrase, replacement in self.phrase_replacements.items():
            text = re.sub(re.escape(phrase), replacement, text, flags=re.IGNORECASE)
        return text
    
    def _simplify_words(self, text):
        words = word_tokenize(text)
        simplified = []
        for word in words:
            lower_word = word.lower()
            if lower_word in self.word_replacements:
                replacement = self.word_replacements[lower_word]
                if word.istitle():
                    replacement = replacement.title()
                simplified.append(replacement)
            else:
                simplified.append(word)
        return ' '.join(simplified)
    
    def _improve_punctuation(self, text):
        text = re.sub(r'\s([,.!?;])', r'\1', text)
        text = re.sub(r',\s*and', ' and', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.\s+\.', '.', text)
        return text.strip()
    
    def _correct_spelling(self, text):
        blob = TextBlob(text)
        corrected = str(blob.correct())
        # Additional check to ensure we don't make things worse
        if len(word_tokenize(text)) == len(word_tokenize(corrected)):
            return corrected
        return text
    
    def paraphrase(self, text):
        simplified = self._simplify_phrase(text)
        simplified = self._simplify_words(simplified)
        simplified = self._improve_punctuation(simplified)
        simplified = self._correct_spelling(simplified)
        
        return {
            'original': text,
            'paraphrased': simplified,
            'improvement_percentage': self._calculate_improvement(text, simplified)
        }
    
    def _calculate_improvement(self, original, paraphrased):
        orig_words = word_tokenize(original)
        para_words = word_tokenize(paraphrased)
        
        orig_complex = sum(1 for word in orig_words if word.lower() in self.word_replacements or 
                          any(phrase.lower() in original.lower() for phrase in self.phrase_replacements))
        
        para_complex = sum(1 for word in para_words if word.lower() in self.word_replacements or 
                          any(phrase.lower() in paraphrased.lower() for phrase in self.phrase_replacements))
        
        if orig_complex == 0:
            return 100
        
        improvement = ((orig_complex - para_complex) / orig_complex) * 100
        return max(0, min(100, round(improvement)))

grammar_checker = GrammarChecker()
paraphraser = PlainEnglishParaphraser()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_grammar', methods=['POST'])
def analyze_grammar():
    try:
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
            
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        result = grammar_checker.mark_grammar_errors(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/paraphrase', methods=['POST'])
def paraphrase_text():
    try:
        if request.is_json:
            data = request.get_json()
            text = data.get('text', '')
        else:
            text = request.form.get('text', '')
            
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        result = paraphraser.paraphrase(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)