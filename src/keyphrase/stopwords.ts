// english stopword list (~175 words) for keyphrase filtering.
// kept intentionally compact — covers determiners, prepositions,
// pronouns, auxiliaries, and common adverbs.
export const STOPWORDS: ReadonlySet<string> = new Set([
  // determiners
  "a", "an", "the", "this", "that", "these", "those",
  "my", "your", "his", "her", "its", "our", "their",
  "some", "any", "no", "every", "each", "all", "both",
  "few", "more", "most", "other", "such", "what", "which",

  // pronouns
  "i", "me", "we", "us", "you", "he", "him", "she",
  "it", "they", "them", "myself", "yourself", "himself",
  "herself", "itself", "ourselves", "themselves",

  // prepositions
  "in", "on", "at", "to", "for", "with", "from", "by",
  "about", "as", "into", "through", "during", "before",
  "after", "above", "below", "between", "under", "over",
  "out", "up", "down", "off", "of", "against", "along",
  "around", "upon", "toward", "towards", "without", "within",

  // conjunctions
  "and", "but", "or", "nor", "so", "yet", "if", "then",
  "than", "because", "while", "although", "though",
  "whether", "either", "neither", "unless", "since",

  // auxiliaries / modals
  "is", "am", "are", "was", "were", "be", "been", "being",
  "have", "has", "had", "having", "do", "does", "did",
  "will", "would", "shall", "should", "may", "might",
  "can", "could", "must",

  // common verbs
  "get", "got", "make", "made", "go", "went", "gone",
  "come", "came", "take", "took", "taken", "give", "gave",
  "say", "said", "tell", "told", "know", "knew", "known",
  "see", "saw", "seen", "think", "thought", "let",

  // adverbs / misc
  "not", "very", "just", "also", "now", "here", "there",
  "when", "where", "how", "why", "who", "whom", "whose",
  "only", "still", "already", "even", "much", "many",
  "well", "back", "too", "quite", "rather", "enough",
  "however", "therefore", "thus", "hence", "otherwise",
  "instead", "anyway", "besides", "moreover", "furthermore",
  "meanwhile", "nevertheless", "nonetheless",

  // misc function words
  "like", "one", "ones", "own", "same", "able", "else",
  "ever", "never", "always", "often", "sometimes",
  "really", "again", "once", "twice", "perhaps", "maybe",
  "almost", "already", "certainly", "probably",
]);
