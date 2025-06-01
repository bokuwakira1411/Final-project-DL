# LLM-based applications with zero-shot and few-shot prompting (ứng dụng LLM cho 5 bài toán: hỏi đáp, phân loại, tính toán, suy luận, khác; áp dụng và so sánh các prompt khác nhau, so sánh zero-shot và few-shot prompts)
Mô tả tóm tắt: Thiết kế prompt và chạy thử nghiệm, so sánh trên 2 mô hình nhẹ 
**mistral-7b-instruct-v0.1.Q4_K_M.gguf**: mô hình thiên về tính toán, suy luận và **flan-T5-xl**: mô hình thiên về những tác vụ phân loại, hỏi đáp
- Các file thử nghiệm chạy hàng loạt trên data được chuẩn bị sẵn và các file chạy chính gồm:
  - **basic_prompt_design_for-flan-T5-xl.ipynb, advanced_prompt_design_for-flan-T5-xl.ipynb**: chạy thử nghiệm (basic prompt techniques) và (advanced prompt techniques) trên flan-T5-xl
      Link kaggle: https://www.kaggle.com/code/bokuwakira/basic-prompt-designs
  - **basic_prompt_design_for-Mistral.ipynb, advanced_prompt_design_for-Mistral.ipynb**: chạy thử nghiệm (basic prompt techniques) và (advanced prompt techniques) trên mistral-7b-instruct-v0.1.Q4_K_M.gguf
  - **Data** : dữ liệu cho 5 tasks hỏi đáp, phân loại, tính toán, suy luận, khác
  - **src/basic_prompt_design_for-flan-T5-xl.py**: thiết kế basic prompt cho flan-T5-xl trên pycharm
  - **src/main, src/demo**
