import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
# import torch
if torch.cuda.is_available():          
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
    model = T5ForConditionalGeneration.from_pretrained("NlpHUST/t5-small-vi-summarization")
tokenizer = T5Tokenizer.from_pretrained("NlpHUST/t5-small-vi-summarization")
model.to(device)
src = "Dự án khu phức hợp, trung tâm thương mại 6,8 ha ở tứ giác Nguyễn Cư Trinh (Mả Lạng), quận 1 bị chính quyền TP HCM thu hồi, sau 16 năm không triển khai.Nội dung được đề cập trong thông báo kết luận của Ban cán sự đảng UBND TP HCM tại cuộc họp mới đây. Sở Kế hoạch và Đầu tư cùng Sở Tư pháp được giao soạn công văn để từ chối nhà đầu tư dự án là Công ty TNHH Tập đoàn Bitexco vì không có cơ sở xem xét đề xuất tiếp tục thực hiện.Cùng với đó, dự án xây mới Bệnh viện Đa khoa Sài Gòn để nhường khu đất hiện hữu (tại 125 Lê Lợi) cho Bitexco làm tổ hợp cao ốc văn phòng thương mại, khách sạn 5 sao cũng tạm dừng.Khu Mả Lạng, hay còn gọi là tứ giác Nguyễn Cư Trinh, giới hạn bởi 4 tuyến đường Nguyễn Trãi - Cống Quỳnh - Trần Đình Xu - Nguyễn Cư Trinh. Nơi này trước 1975 là nghĩa địa sau đó được thành phố cho di dời. Về sau, nhiều người đến sinh sống và trở thành khu dân cư tại trung tâm quận 1. Trong khu vực có hơn 530 nhà dưới 20 m2, chủ yếu là siêu nhỏ, xuống cấp.Từ năm 2000, TP HCM có chủ trương giải tỏa khu Mả Lạng với tổng diện tích 6,8 ha nhằm chỉnh trang đô thị và giao Tổng Công ty Địa ốc Sài Gòn triển khai, nhưng không làm được. Năm 2007, dự án được chuyển cho Tập đoàn Bitexco để thực hiện khu phức hợp khách sạn, cao ốc văn phòng, trung tâm thương mại. Tổng số nhà phải giải tỏa là 1.424 căn. Dự kiến việc di dời, tái định cư bắt đầu từ tháng 6/2018. Tuy nhiên, dự án tiếp tục bị treo đến nay.Do khó thu hồi đất, từ giữa năm 2022, UBND TP HCM đã giao các đơn vị nghiên cứu cơ sở pháp lý, tham mưu chấm dứt dự án."
tokenized_text = tokenizer.encode(src, return_tensors="pt").to(device)
model.eval()
summary_ids = model.generate(
                    tokenized_text,
                    max_length=256,                    
                    num_beams=5,
                    repetition_penalty=2.5,                    
                    length_penalty=1.0,                    
                    early_stopping=True
                )
output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(output)
