from openai import OpenAI

class AIAdvisor:
    def __init__(self, data, key, model):
        self.data = data
        self.openai_api_key = key
        self.model = model
        self.client = OpenAI(api_key=self.openai_api_key)
        self.response = []

    def chunk_text(self, text, max_tokens=15000):
        words = text.split()
        chunks = []
        current_chunk = []

        for word in words:
            if len(current_chunk) + len(word) <= max_tokens:
                current_chunk.append(word)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks


    def get_answer_from_chatgpt(self, question, context):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": context},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content

    def get_advice(self, question):
        context_chunks = self.chunk_text(self.data)
        advices = [self.get_answer_from_chatgpt(question, chunk) for chunk in context_chunks]
        advice = "\n".join(advices)

        # Save the question and advice to self.response for finetuning
        self.response.append({
            'question': question,
            'advice': advice
        })

        return advice



    def put_finetuning(self):
        response = openai.File.create(
            file=open("training_data.jsonl"),
            purpose='fine-tune'
        )
        file_id = response['id']

        response = openai.FineTune.create(
            training_file=file_id,
            model=self.model
        )
        self.fine_tune_id = response['id']