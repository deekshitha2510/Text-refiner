import os
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
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.common_words = set(nltk.corpus.words.words()) if hasattr(nltk.corpus, 'words') else set()
    
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
                logger.info(f"NLTK resource {resource} already available")
            except LookupError:
                logger.info(f"Downloading NLTK resource {resource}")
                nltk.download(resource, quiet=True)
    
    def mark_grammar_errors(self, text, max_length=1000):
        if len(text) > max_length:
            raise ValueError(f"Text exceeds maximum length of {max_length} characters")
            
        try:
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
        except Exception as e:
            logger.error(f"Error in mark_grammar_errors: {str(e)}")
            raise
    
    def _check_spelling(self, word):
        if not word.isalpha() or len(word) <= 1:
            return None
            
        if word[0].isupper():
            return None
            
        if word.lower() in self.common_words:
            return None
            
        if wordnet.synsets(word.lower()):
            return None
            
        word_obj = Word(word.lower())
        suggestions = word_obj.spellcheck()
        
        if suggestions and suggestions[0][0].lower() != word.lower() and suggestions[0][1] > 0.7:
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
                if word.get('spelling_issues', {}).get('incorrect'):
                    spelling_errors += 1
        
        return {
            'parts_of_speech': dict(pos_counts),
            'spelling_errors': spelling_errors,
            'total_sentences': len(analysis),
            'total_words': sum(len(s['analysis']) for s in analysis)
        }

class PlainEnglishParaphraser:
    def __init__(self):
        self._download_nltk_resources()
        self.phrase_replacements = {
            r'in\s+spite\s+of\s+the\s+fact\s+that': 'although',
            r'at\s+this\s+point\s+in\s+time': 'now',
            r'the\s+majority\s+of': 'most',
            r'are\s+in\s+agreement': 'agree',
            r'not\s+too\s+distant\s+future': 'soon',
            r'may\s+be\s+eliminated\s+altogether': 'may be cut',
            r'there\s+is\s+a\s+solemn\s+danger': 'there is a risk',
            r'professional\s+nurses': 'school nurses',
            r'are\s+essential': 'are needed'
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
            ('omw-1.4', 'corpora')
        ]
        for resource, package in resources:
            try:
                nltk.data.find(f'{package}/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)
    
    def _simplify_phrase(self, text):
        for phrase, replacement in self.phrase_replacements.items():
            text = re.sub(phrase, replacement, text, flags=re.IGNORECASE)
        return text
    
    def _simplify_words(self, text):
        words = word_tokenize(text)
        simplified = []
        for word in words:
            lower_word = word.lower()
            if lower_word in self.word_replacements:
                replacement = self.word_replacements[lower_word]
                simplified.append(replacement.capitalize() if word.istitle() else replacement)
            else:
                simplified.append(word)
        return ' '.join(simplified)
    
    def _improve_punctuation(self, text):
        text = re.sub(r'\s([,.!?;])', r'\1', text)
        text = re.sub(r',\s*and', ' and', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def paraphrase(self, text, max_length=1000):
        if len(text) > max_length:
            raise ValueError(f"Text exceeds maximum length of {max_length} characters")
            
        try:
            simplified = self._simplify_phrase(text)
            simplified = self._simplify_words(simplified)
            simplified = self._improve_punctuation(simplified)
            
            return {
                'original': text,
                'paraphrased': simplified,
                'improvement_percentage': self._calculate_improvement(text, simplified)
            }
        except Exception as e:
            logger.error(f"Error in paraphrase: {str(e)}")
            raise
    
    def _calculate_improvement(self, original, paraphrased):
        orig_words = word_tokenize(original.lower())
        para_words = word_tokenize(paraphrased.lower())
        
        orig_complex = sum(1 for word in orig_words if word in self.word_replacements or 
                          any(re.search(phrase, original.lower()) for phrase in self.phrase_replacements))
        
        para_complex = sum(1 for word in para_words if word in self.word_replacements or 
                          any(re.search(phrase, paraphrased.lower()) for phrase in self.phrase_replacements))
        
        if orig_complex == 0:
            return 100
        
        improvement = ((orig_complex - para_complex) / orig_complex) * 100
        return max(0, min(100, round(improvement)))

# Initialize components
grammar_checker = GrammarChecker()
paraphraser = PlainEnglishParaphraser()

@app.route('/')
def home():
    return jsonify({
        'status': 'active',
        'endpoints': {
            '/analyze_grammar': 'POST with {"text": "your text"}',
            '/paraphrase': 'POST with {"text": "your text"}'
        }
    })

@app.route('/analyze_grammar', methods=['POST'])
def analyze_grammar():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
            
        result = grammar_checker.mark_grammar_errors(text)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Grammar analysis error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/paraphrase', methods=['POST'])
def paraphrase_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
            
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
            
        result = paraphraser.paraphrase(text)
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Paraphrasing error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)