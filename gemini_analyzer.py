import os
import json
import logging
import random
from typing import Dict, List, Optional
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Dict, Any


class SegmentAnalysis(BaseModel):
    engagement_score: float
    emotion_score: float
    viral_potential: float
    quotability: float
    emotions: List[str]
    keywords: List[str]
    reason: str


class VideoMetadata(BaseModel):
    title: str
    description: str
    tags: List[str]


class GeminiAnalyzer:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.use_fallback_only = False  # Enable Gemini AI for viral content generation
        self.api_keys = []
        self.current_key_index = 0

        # Collect API keys and initialize client
        self._collect_api_keys()
        self.logger.info(f"Found {len(self.api_keys)} Gemini API key(s)")

        if self.api_keys:
            self._initialize_client()
            self.logger.info("Gemini AI enabled for viral content generation")
        else:
            self.use_fallback_only = True
            self.logger.warning(
                "No Gemini API keys found, using fallback methods")

    def _collect_api_keys(self):
        """Collect all available Gemini API keys from environment"""
        # Primary key
        primary_key = os.environ.get("GEMINI_API_KEY")
        if primary_key:
            self.api_keys.append(primary_key)

        # Backup keys
        for i in range(1, 5):  # Support up to 4 backup keys
            backup_key = os.environ.get(f"GEMINI_API_KEY_{i}")
            if backup_key:
                self.api_keys.append(backup_key)

        self.logger.info(f"Found {len(self.api_keys)} Gemini API key(s)")

    def _initialize_client(self):
        """Initialize client with current API key"""
        if self.current_key_index < len(self.api_keys):
            try:
                api_key = self.api_keys[self.current_key_index]
                self.client = genai.Client(api_key=api_key)
                self.logger.info(
                    f"Gemini client initialized with API key #{self.current_key_index + 1}"
                )
                return True
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize Gemini client with key #{self.current_key_index + 1}: {e}"
                )
                return False
        return False

    def _switch_to_next_key(self):
        """Switch to next available API key"""
        self.current_key_index += 1
        if self.current_key_index < len(self.api_keys):
            self.logger.info(
                f"Switching to backup API key #{self.current_key_index + 1}")
            if self._initialize_client():
                return True

        # No more keys available
        self.logger.warning(
            "All Gemini API keys exhausted, switching to fallback mode")
        self.use_fallback_only = True
        self.client = None
        return False

    def _handle_api_error(self, error_msg: str):
        """Handle API errors and attempt key switching"""
        # Check for quota exceeded or rate limit errors
        if any(indicator in error_msg.lower() for indicator in
               ["429", "resource_exhausted", "quota", "rate limit"]):
            self.logger.warning(f"API quota/rate limit hit: {error_msg}")
            return self._switch_to_next_key()

        # For other errors, log but don't switch keys
        self.logger.error(f"API error: {error_msg}")
        return False

    def analyze_segment(self, text: str) -> Dict[str, Any]:
        """Analyze a text segment for engagement and viral potential using Gemini - OPTIMIZED"""
        # Check if we should use fallback only
        if self.use_fallback_only or not self.client:
            self.logger.info(
                "Using fallback analysis (no Gemini API available)")
            return self._fallback_analysis(text)

        # Optimize text length for faster processing
        if len(text) > 2000:  # Limit text length for faster AI processing
            text = text[:2000] + "..."

        try:
            system_prompt = """You are an expert content analyst specializing in viral social media content and YouTube Shorts.

            Analyze the given text segment for its potential to create engaging short-form video content.

            Consider these factors:
            - Engagement Score (0.0-1.0): How likely this content is to engage viewers
            - Emotion Score (0.0-1.0): Emotional impact and intensity
            - Viral Potential (0.0-1.0): Likelihood to be shared and go viral
            - Quotability (0.0-1.0): How memorable and quotable the content is
            - Emotions: List of emotions detected (humor, surprise, excitement, inspiration, etc.)
            - Keywords: Important keywords that make this content engaging
            - Reason: Brief explanation of why this segment is engaging

            Focus on content that has:
            - Strong emotional hooks
            - Surprising or unexpected elements
            - Humor or entertainment value
            - Inspirational or motivational content
            - Controversial or debate-worthy topics
            - Clear storytelling elements
            - Quotable phrases or moments"""

            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=
                                f"Analyze this content segment for YouTube Shorts potential:\n\n{text}"
                            )
                        ])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=SegmentAnalysis,
                ),
            )

            if response.text:
                result = json.loads(response.text)
                return {
                    'engagement_score':
                    max(0.0, min(1.0, result.get('engagement_score', 0.5))),
                    'emotion_score':
                    max(0.0, min(1.0, result.get('emotion_score', 0.5))),
                    'viral_potential':
                    max(0.0, min(1.0, result.get('viral_potential', 0.5))),
                    'quotability':
                    max(0.0, min(1.0, result.get('quotability', 0.5))),
                    'emotions':
                    result.get('emotions', [])[:5],  # Limit to 5 emotions
                    'keywords':
                    result.get('keywords', [])[:10],  # Limit to 10 keywords
                    'reason':
                    result.get('reason',
                               'Content has potential for engagement')[:500]
                }
            else:
                raise Exception("Empty response from Gemini")

        except Exception as e:
            error_msg = str(e)

            # Try to switch to next API key if error is quota-related
            if self._handle_api_error(
                    error_msg) and not self.use_fallback_only:
                # Retry with new key
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=[
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part(
                                        text=
                                        f"Analyze this content segment for YouTube Shorts potential:\n\n{text}"
                                    )
                                ])
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=SegmentAnalysis,
                        ),
                    )

                    if response.text:
                        result = json.loads(response.text)
                        return {
                            'engagement_score':
                            max(0.0,
                                min(1.0, result.get('engagement_score', 0.5))),
                            'emotion_score':
                            max(0.0, min(1.0, result.get('emotion_score',
                                                         0.5))),
                            'viral_potential':
                            max(0.0,
                                min(1.0, result.get('viral_potential', 0.5))),
                            'quotability':
                            max(0.0, min(1.0, result.get('quotability', 0.5))),
                            'emotions':
                            result.get('emotions', [])[:5],
                            'keywords':
                            result.get('keywords', [])[:10],
                            'reason':
                            result.get(
                                'reason',
                                'Content has potential for engagement')[:500]
                        }
                except Exception as retry_e:
                    self.logger.error(
                        f"Retry with backup key failed: {retry_e}")

            # Fallback analysis
            return self._fallback_analysis(text)

    def _fallback_analysis(self, text: str) -> Dict[str, Any]:
        """Enhanced fallback analysis when Gemini is unavailable"""
        text_lower = text.lower()
        words = text.split()

        # Enhanced keyword categories
        engagement_keywords = [
            'amazing', 'incredible', 'wow', 'shocking', 'unbelievable',
            'funny', 'hilarious', 'awesome', 'fantastic', 'mind-blowing',
            'crazy', 'insane', 'epic', 'legendary'
        ]
        emotion_keywords = [
            'love', 'hate', 'excited', 'surprised', 'happy', 'angry', 'scared',
            'thrilled', 'disappointed', 'frustrated', 'overwhelmed',
            'passionate', 'emotional', 'heartwarming'
        ]
        viral_keywords = [
            'viral', 'trending', 'share', 'like', 'subscribe', 'follow',
            'must-see', 'breaking', 'exclusive', 'revealed', 'secret',
            'exposed', 'truth', 'shocking'
        ]
        quotable_keywords = [
            'said', 'quote', 'tells', 'explains', 'reveals', 'admits',
            'confesses', 'announces'
        ]

        # Calculate scores based on keyword presence
        engagement_score = min(
            1.0,
            sum(1
                for word in engagement_keywords if word in text_lower) * 0.15)
        emotion_score = min(
            1.0,
            sum(1 for word in emotion_keywords if word in text_lower) * 0.15)
        viral_score = min(
            1.0,
            sum(1 for word in viral_keywords if word in text_lower) * 0.2)
        quotability_score = min(
            1.0,
            sum(1 for word in quotable_keywords if word in text_lower) * 0.2)

        # Length-based scoring (optimal length for shorts)
        text_length = len(words)
        if 20 <= text_length <= 50:  # Optimal length for short clips
            length_bonus = 0.2
        elif 10 <= text_length <= 80:  # Good length
            length_bonus = 0.1
        else:
            length_bonus = 0.0

        # Add length bonus to all scores
        engagement_score = min(1.0, engagement_score + length_bonus)
        emotion_score = min(1.0, emotion_score + length_bonus)
        viral_score = min(1.0, viral_score + length_bonus)
        quotability_score = min(1.0, quotability_score + length_bonus)

        # Ensure minimum scores for content viability
        engagement_score = max(0.4, engagement_score)
        emotion_score = max(0.3, emotion_score)
        viral_score = max(0.3, viral_score)
        quotability_score = max(0.2, quotability_score)

        # Detect emotions based on keywords
        detected_emotions = []
        if any(word in text_lower
               for word in ['funny', 'hilarious', 'joke', 'laugh']):
            detected_emotions.append('humor')
        if any(word in text_lower
               for word in ['shocking', 'surprised', 'unexpected']):
            detected_emotions.append('surprise')
        if any(word in text_lower
               for word in ['love', 'heartwarming', 'beautiful']):
            detected_emotions.append('inspiration')
        if any(word in text_lower for word in ['angry', 'frustrated', 'hate']):
            detected_emotions.append('controversy')
        if not detected_emotions:
            detected_emotions = ['general']

        # Extract meaningful keywords (longer words, excluding common words)
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'a', 'an'
        }
        keywords = [
            word for word in words
            if len(word) > 3 and word.lower() not in common_words
        ][:8]

        return {
            'engagement_score':
            engagement_score,
            'emotion_score':
            emotion_score,
            'viral_potential':
            viral_score,
            'quotability':
            quotability_score,
            'emotions':
            detected_emotions[:5],
            'keywords':
            keywords,
            'reason':
            f'Fallback analysis: {len(words)} words, detected {", ".join(detected_emotions)} content'
        }

    def _enhance_description_to_500_words(self, description: str,
                                          segment_text: str,
                                          original_title: str) -> str:
        """Enhance description to approximately 500 words for viral potential"""
        if len(description.split()) >= 400:
            return description

        # Extract key elements from the segment
        words = segment_text.split()
        key_points = []

        # Add context about the original video
        enhanced_desc = f"🔥 VIRAL MOMENT ALERT! 🔥\n\n"
        enhanced_desc += f"This INCREDIBLE clip from '{original_title}' is absolutely MIND-BLOWING! "
        enhanced_desc += f"What you're about to see will leave you SPEECHLESS and wondering how this even happened! "

        # Add the original description
        enhanced_desc += f"\n\n{description}\n\n"

        # Add engaging context
        enhanced_desc += f"But wait, there's MORE! This isn't just any ordinary moment - this is the kind of content that "
        enhanced_desc += f"BREAKS the internet and gets shared millions of times! The way everything unfolds is absolutely "
        enhanced_desc += f"INSANE and you need to see it to believe it!\n\n"

        # Add emotional hooks
        enhanced_desc += f"💥 Why is this going VIRAL?\n"
        enhanced_desc += f"✨ The timing is PERFECT\n"
        enhanced_desc += f"🤯 The reaction is PRICELESS\n"
        enhanced_desc += f"🔥 This moment is LEGENDARY\n\n"

        # Add engagement prompts
        enhanced_desc += f"👆 SMASH that LIKE button if this gave you CHILLS!\n"
        enhanced_desc += f"💬 COMMENT below what you think about this CRAZY moment!\n"
        enhanced_desc += f"🔔 SUBSCRIBE for more VIRAL content like this!\n"
        enhanced_desc += f"📱 SHARE this with everyone - they NEED to see this!\n\n"

        # Add trending hashtags
        enhanced_desc += f"#Shorts #Viral #Trending #Fyp #Amazing #Shocking #MustWatch #Unbelievable #MindBlown #Epic "
        enhanced_desc += f"#Insane #Crazy #OMG #NoWay #Incredible #Legendary #Speechless #Chills #Goosebumps #WOW\n\n"

        # Add questions for engagement
        enhanced_desc += f"🤔 What did you think when you saw this?\n"
        enhanced_desc += f"😱 Have you ever experienced something like this?\n"
        enhanced_desc += f"🔥 What would YOUR reaction be?\n\n"

        # Add urgency and FOMO
        enhanced_desc += f"⚡ This is the moment EVERYONE is talking about!\n"
        enhanced_desc += f"🚨 Don't miss out on the HOTTEST viral content!\n"
        enhanced_desc += f"⏰ Watch before it gets even MORE popular!\n\n"

        # Final call to action
        enhanced_desc += f"DROP a 🔥 in the comments if this was AMAZING! Let's get this to 1 MILLION views!"

        return enhanced_desc

    def generate_metadata(self,
                          segment_text: str,
                          original_title: str,
                          language: str = "hinglish",
                          segment_index: int = 0,
                          total_segments: int = 1) -> Dict[str, Any]:
        """Generate unique title, description, and tags for a video short using Gemini with enhanced context"""
        # Check if we should use fallback only
        if self.use_fallback_only or not self.client:
            self.logger.info(
                "Using fallback metadata generation (no Gemini API available)")
            return self._fallback_metadata(segment_text, original_title,
                                           language, segment_index)

        try:
            # Extract key phrases and emotions from segment for uniqueness
            unique_context = self._extract_unique_context(segment_text)

            if language.lower() == "hindi":
                system_prompt = """आप एक वायरल YouTube Shorts विशेषज्ञ हैं जो करोड़ों व्यूज पाने वाला UNIQUE कंटेंट बनाते हैं।

हर Short के लिए COMPLETELY UNIQUE मेटाडेटा बनाएं जो अधिकतम एंगेजमेंट और व्यूज लाए।

UNIQUE वायरल टाइटल नियम (अधिकतम 60 अक्षर):
- हर टाइटल TOTALLY DIFFERENT होना चाहिए
- Content के specific moments के base पर बनाएं
- Power emojis: 😱💥🔥⚡🤯💯🚀✨🎯🎊
- Curiosity hooks: "इसके बाद जो हुआ...", "कोई नहीं जानता कि...", "सच्चाई यह है..."
- Numbers और specific details use करें

UNIQUE वायरल विवरण नियम (500+ शब्द):
- हर description completely different story tell करे
- Segment के specific content को highlight करें  
- 20+ strategic hashtags with variety
- Multiple engaging questions और hooks
- Emotional storytelling with CAPS emphasis
- Different call-to-actions हर बार

UNIQUE वायरल टैग्स (25+ टैग्स):
- Core tags: #Shorts #Viral #Trending #Fyp
- Content-specific tags based on segment
- Emotional variety: #चौंकाने_वाला #अविश्वसनीय #रोमांचक #दिलचस्प
- Regional tags: #हिंदी #भारत #देसी #इंडिया"""
            else:  # Hinglish
                system_prompt = """You are a viral YouTube Shorts expert who creates UNIQUE content that gets CRORES of views.

Create COMPLETELY UNIQUE metadata for each Short that maximizes engagement and views using Hindi-English mix.

UNIQUE VIRAL TITLE RULES (max 60 characters):
- Every title must be TOTALLY DIFFERENT and UNIQUE
- Base it on specific moments from the content segment
- Power emojis: 😱💥🔥⚡🤯💯🚀✨🎯🎊🔮💎🌟
- Hinglish viral hooks: "Bhai ye dekho kya hua!", "Itna CRAZY moment!", "Yaar ye kaise possible hai!"
- Use specific numbers, names, or details from content
- Create curiosity gaps: "Iske baad jo hua...", "Sabko pata hona chahiye...", "Ye secret..."

UNIQUE VIRAL DESCRIPTION RULES (500+ words):
- Every description tells a COMPLETELY DIFFERENT story
- Highlight specific content moments from this segment
- Mix Hindi-English naturally throughout
- 20+ strategic hashtags with content variety
- Multiple engaging questions specific to this segment
- Emotional storytelling with strategic CAPS
- Different call-to-actions every time
- Create FOMO specific to this content

UNIQUE VIRAL TAGS (25+ tags):
- Core viral tags: #Shorts #Viral #Trending #Fyp #Hinglish #India
- Content-specific tags based on segment analysis  
- Emotional variety: #Shocking #Amazing #Unbelievable #MindBlown #Insane #Epic #Crazy #OMG
- Cultural tags: #Desi #Bollywood #IndianContent #देसी #भारत
- Trending variations each time"""

            if language.lower() == "hindi":
                prompt = f"""इस specific segment के लिए UNIQUE वायरल मेटाडेटा बनाएं:

मूल वीडियो: {original_title}
सेगमेंट #{segment_index + 1} (कुल {total_segments} में से): {segment_text}

UNIQUE कंटेक्स्ट: {unique_context}

REQUIREMENTS:
- टाइटल में इस segment के specific moment को highlight करें
- Description में इस particular content के बारे में unique story बताएं
- हर Short के लिए different approach use करें
- Specific details और moments का use करें जो सिर्फ इस segment में हैं
- Completely unique hashtags और hooks create करें"""
            else:  # Hinglish
                prompt = f"""Create UNIQUE viral metadata for this specific segment - Hinglish style:

ORIGINAL VIDEO: {original_title}
SEGMENT #{segment_index + 1} (out of {total_segments}): {segment_text}

UNIQUE CONTEXT: {unique_context}

REQUIREMENTS:
- Title should highlight specific moment from THIS segment only
- Description should tell unique story about THIS particular content
- Use different approach for every Short
- Include specific details/moments that are unique to this segment
- Create completely unique hashtags and hooks for this content
- Make it sound like a different person is describing this amazing moment"""

            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Content(role="user", parts=[types.Part(text=prompt)])
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=VideoMetadata,
                ),
            )

            if response.text:
                result = json.loads(response.text)

                # Ensure description is around 500 words
                description = result.get('description', '')
                if len(description.split()) < 400:
                    description = self._enhance_description_to_500_words(
                        description, segment_text, original_title)

                return {
                    'title':
                    result.get(
                        'title',
                        f"SHOCKING Moment from {original_title} GOES VIRAL!")
                    [:60],
                    'description':
                    description,
                    'tags':
                    result.get('tags', [
                        '#Shorts', '#Viral', '#Trending', '#Fyp', '#Amazing',
                        '#Shocking', '#MustWatch', '#Unbelievable'
                    ])[:25]
                }
            else:
                raise Exception("Empty response from Gemini")

        except Exception as e:
            error_msg = str(e)

            # Try to switch to next API key if error is quota-related
            if self._handle_api_error(
                    error_msg) and not self.use_fallback_only:
                # Retry with new key
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-pro",
                        contents=[
                            types.Content(role="user",
                                          parts=[types.Part(text=prompt)])
                        ],
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            response_mime_type="application/json",
                            response_schema=VideoMetadata,
                        ),
                    )

                    if response.text:
                        result = json.loads(response.text)
                        return {
                            'title':
                            result.get(
                                'title',
                                f"Viral Moment from {original_title}")[:100],
                            'description':
                            result.get(
                                'description',
                                f"Amazing clip from {original_title}\n\n#Shorts #Viral #Trending"
                            ),
                            'tags':
                            result.get('tags', [
                                'shorts', 'viral', 'trending', 'entertainment'
                            ])[:25]
                        }
                except Exception as retry_e:
                    self.logger.error(
                        f"Retry with backup key failed: {retry_e}")

            return self._fallback_metadata(segment_text, original_title,
                                           language, segment_index)

    def _extract_unique_context(self, segment_text: str) -> str:
        """Extract unique context elements from segment for title generation"""
        words = segment_text.split()
        text_lower = segment_text.lower()

        # Extract unique elements
        unique_elements = []

        # Look for names, numbers, specific actions
        import re
        numbers = re.findall(r'\b\d+\b', segment_text)
        if numbers:
            unique_elements.append(f"numbers: {', '.join(numbers[:3])}")

        # Look for action words
        action_words = [
            'said', 'tells', 'shows', 'reveals', 'does', 'makes', 'creates',
            'breaks', 'wins', 'loses', 'finds', 'discovers'
        ]
        found_actions = [word for word in action_words if word in text_lower]
        if found_actions:
            unique_elements.append(f"actions: {', '.join(found_actions[:3])}")

        # Look for emotional indicators
        emotions = [
            'funny', 'shocking', 'amazing', 'crazy', 'incredible',
            'unbelievable', 'awesome', 'terrible', 'wonderful'
        ]
        found_emotions = [word for word in emotions if word in text_lower]
        if found_emotions:
            unique_elements.append(
                f"emotions: {', '.join(found_emotions[:3])}")

        # Look for objects/things
        objects = [
            'video', 'music', 'song', 'movie', 'book', 'game', 'food', 'car',
            'house', 'phone', 'computer'
        ]
        found_objects = [word for word in objects if word in text_lower]
        if found_objects:
            unique_elements.append(f"objects: {', '.join(found_objects[:3])}")

        # Get key phrases (2-3 word combinations)
        key_phrases = []
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i + 1]) > 3:
                phrase = f"{words[i]} {words[i+1]}"
                key_phrases.append(phrase)

        if key_phrases:
            unique_elements.append(f"phrases: {', '.join(key_phrases[:2])}")

        return " | ".join(
            unique_elements) if unique_elements else "general content"

    def _fallback_metadata(self,
                           segment_text: str,
                           original_title: str,
                           language: str = "hinglish",
                           segment_index: int = 0) -> Dict[str, Any]:
        """Enhanced fallback metadata generation"""
        words = segment_text.split()
        text_lower = segment_text.lower()

        # Extract meaningful keywords (exclude common words)
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'is', 'are', 'was', 'were', 'a', 'an', 'this', 'that'
        }
        key_words = [
            word for word in words
            if len(word) > 3 and word.lower() not in common_words
        ][:5]

        # Generate UNIQUE viral title based on content type, language and segment index
        viral_emojis = [
            "😱", "🔥", "💥", "⚡", "🤯", "💯", "🚀", "✨", "🎯", "🔮", "💎", "🌟"
        ]
        emoji = viral_emojis[segment_index % len(viral_emojis)]

        # Create unique variations for each segment
        if language.lower() == "hindi":
            title_variations = [
                f"{emoji} इसके बाद जो हुआ वो देखिए!",
                f"{emoji} यह {key_words[0] if key_words else 'पल'} चौंका देगा!",
                f"{emoji} कोई नहीं जानता यह राज!",
                f"{emoji} इस वीडियो का सच जानें!",
                f"{emoji} {segment_index + 1}वां हिस्सा वायरल!",
                f"{emoji} देखें क्या हुआ अगला!",
                f"{emoji} यह मोमेंट इतिहास बनेगा!",
                f"{emoji} सबसे क्रेजी पार्ट यहाँ है!"
            ]
        else:  # Hinglish
            title_variations = [
                f"{emoji} Bhai ye dekho kya hua next!",
                f"{emoji} Is {key_words[0] if key_words else 'moment'} ne tod diya!",
                f"{emoji} Yaar itna CRAZY kaise possible!",
                f"{emoji} OMG ye {segment_index + 1}th part INSANE!",
                f"{emoji} Dost ye secret koi nahi jaanta!",
                f"{emoji} Dekho iske baad kya twist!",
                f"{emoji} Bhai ye part sabse VIRAL!",
                f"{emoji} Yaar ye moment LEGENDARY hai!"
            ]

        title = title_variations[segment_index % len(title_variations)][:60]

        # Generate unique viral 500-word description with segment-specific content
        unique_hooks = [
            f"🔥 GUYS ye {segment_index + 1}th part dekhne ke baad aap PAGAL ho jaenge!",
            f"😱 Bhai maine ye dekha aur mera dimag hil gaya!",
            f"💥 Yaar is video ka ye moment sabse INSANE hai!",
            f"🤯 Dekho kaise ye {key_words[0] if key_words else 'scene'} ne internet tod diya!",
            f"⚡ OMG ye part itna VIRAL kyun ho raha hai?",
            f"🚀 Bhai ye {segment_index + 1}th segment mein jo hua wo UNBELIEVABLE!",
            f"✨ Yaar ye moment dekh kar goosebumps aa gaye!",
            f"🎯 Is video ka sabse EPIC part yahi hai!"
        ]

        hook = unique_hooks[segment_index % len(unique_hooks)]
        description = self._enhance_description_to_500_words(
            hook, segment_text, original_title)

        # Generate viral tags with hashtags based on language
        if language.lower() == "hindi":
            viral_tags = [
                '#Shorts', '#Viral', '#Trending', '#Fyp', '#हिंदी', '#भारत',
                '#देसी', '#चौंकाने_वाला'
            ]

            if any(word in text_lower
                   for word in ['funny', 'comedy', 'hilarious']):
                viral_tags.extend(['#मजेदार', '#कॉमेडी', '#हंसी'])
            if any(word in text_lower for word in ['music', 'song', 'dance']):
                viral_tags.extend(['#संगीत', '#गाना', '#नृत्य'])
            if any(word in text_lower for word in ['food', 'cooking']):
                viral_tags.extend(['#खाना', '#पकाना', '#रेसिपी'])

            viral_tags.extend(['#अविश्वसनीय', '#महान', '#पागल', '#ओएमजी'])
        else:  # Hinglish
            viral_tags = [
                '#Shorts', '#Viral', '#Trending', '#Fyp', '#Hinglish',
                '#India', '#Desi', '#MustWatch'
            ]

            if any(word in text_lower
                   for word in ['funny', 'comedy', 'hilarious']):
                viral_tags.extend(['#Funny', '#Comedy', '#LOL', '#Mazedaar'])
            if any(word in text_lower for word in ['music', 'song', 'dance']):
                viral_tags.extend(['#Music', '#Song', '#Dance', '#Bollywood'])
            if any(word in text_lower for word in ['food', 'cooking']):
                viral_tags.extend(['#Food', '#Indian', '#Recipe', '#Khana'])

            viral_tags.extend(
                ['#MindBlown', '#Epic', '#Insane', '#OMG', '#Bhai', '#यार'])

        return {
            'title': title,
            'description': description,
            'tags': viral_tags[:25]  # Limit to 25 tags
        }

    def analyze_video_file(self, video_path: str) -> Dict[str, Any]:
        """Analyze video file directly with Gemini vision capabilities"""
        # Check if we should use fallback only
        if self.use_fallback_only or not self.client:
            self.logger.info(
                "Video file analysis not available (no Gemini API)")
            return {
                'analysis':
                'Video analysis not available - using audio transcript analysis instead'
            }

        try:
            with open(video_path, "rb") as f:
                video_bytes = f.read()

            response = self.client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    types.Part.from_bytes(
                        data=video_bytes,
                        mime_type="video/mp4",
                    ),
                    "Analyze this video for engaging moments, emotional highlights, and viral potential. "
                    "Identify the most interesting segments that would work well as YouTube Shorts."
                ],
            )

            return {
                'analysis':
                response.text if response.text else 'No analysis available'
            }

        except Exception as e:
            self.logger.error(f"Video file analysis failed: {e}")
            return {'analysis': 'Video analysis not available'}

    def _get_default_analysis(self):
        """Return default analysis scores when AI analysis fails"""
        return {
            'engagement_score': 0.5,
            'emotion_score': 0.5,
            'viral_potential': 0.5,
            'quotability': 0.5,
            'emotions': ['neutral'],
            'keywords': ['content'],
            'content_type': 'general',
            'visual_style': 'clean',
            'reason': 'Default analysis - AI processing unavailable'
        }