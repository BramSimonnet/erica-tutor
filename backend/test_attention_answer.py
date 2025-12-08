"""Test attention mechanism answer with code examples"""
from graph.graphrag_query import answer_with_graphrag

# Test with the attention question
result = answer_with_graphrag(
    'Explain attention mechanisms in transformers',
    top_k_concepts=5,
    min_similarity=0.2,
    include_metadata=True
)

print('=' * 80)
print('ATTENTION MECHANISMS ANSWER')
print('=' * 80)
print()
print(result['answer'])
print()
print('=' * 80)

# Check if code is included
if 'import' in result['answer'] or 'def ' in result['answer'] or '```' in result['answer']:
    print('✓ Code example included!')
else:
    print('✗ No code example found - LLM may need more explicit prompting')

# Save to file
with open('/app/data/attention_answer_test.txt', 'w') as f:
    f.write(result['answer'])

print('\nAnswer saved to: /app/data/attention_answer_test.txt')
