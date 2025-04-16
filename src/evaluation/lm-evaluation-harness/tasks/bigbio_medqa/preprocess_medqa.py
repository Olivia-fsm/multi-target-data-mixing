def doc_to_text(doc) -> str:
    answers = "".join((f"{v['key']}. {v['value']}\n") for v in doc["options"])
    return f"Question: {doc['question']}\n{answers}Answer:"

def doc_to_target(doc) -> str:
  return doc["answer_idx"]
