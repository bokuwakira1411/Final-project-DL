# LLM-based applications with zero-shot and few-shot prompting (ứng dụng LLM cho 5 bài toán: hỏi đáp, phân loại, tính toán, suy luận, khác; áp dụng và so sánh các prompt khác nhau, so sánh zero-shot và few-shot prompts)
Mô tả tóm tắt: Thiết kế prompt và chạy thử nghiệm, so sánh trên 2 mô hình nhẹ
**Microsoft-Phi2**: mô hình thiên về tính toán, suy luận và **flan-T5-xl**: mô hình thiên về những tác vụ phân loại, hỏi đáp
- Các file thử nghiệm chạy hàng loạt trên data được chuẩn bị sẵn và các file chạy chính gồm:
  - **Flan-T5-XL/experiments**: chạy thử nghiệm trên flan-T5-xl
      Link đầy đủ các lượt chạy thử nghiệm kaggle: https://www.kaggle.com/code/bokuwakira/basic-prompt-designs
  - **Microsoft-Phi2/experiments**: chạy thử nghiệm trên Microsoft-Phi2
  - **Data** : dữ liệu cho 5 tasks hỏi đáp, phân loại, tính toán, suy luận, khác
  - **run_Demo**: file chạy chính trên google colab
  - **Final_Project_DL.pdf**: file tổng hợp kết quả chạy
  - **General.py**: file chạy các bài toán khác, phải tự viết prompt trên giao diện
<p align="center">
  <img src="images/Screenshot 2025-06-06 204910.png" alt="Giao diện" width="600"/>
</p>

