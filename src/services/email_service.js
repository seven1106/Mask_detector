require("dotenv").config();
import nodeMailer from "nodemailer";
let sendMailToClient = async (dataSend) => {
  let transporter = nodeMailer.createTransport({
    host: "smtp.gmail.com",
    port: 587,
    secure: false,
    auth: {
      user: process.env.EMAIL_APP,
      pass: process.env.EMAIL_APP_PASSWORD,
    },
  });
  let info = await transporter.sendMail({
    from: '"AMIGO HEALTHCARE" <process.env.EMAIL_APP>',
    to: dataSend.receiverEmail,
    subject: "Thông tin đặt lịch khám bệnh",
    html: `
        <h3>Xin chào ${dataSend.clientName} </h3>
        <p> Bạn nhận được email này vì đã đặt lịch khám bệnh online trên trang web Amigo healthcare. </p>
        <p>Chúng tôi đánh giá cao sự tin tưởng của bạn đối với các dịch vụ chăm sóc sức khỏe của chúng tôi và mong muốn được cung cấp cho bạn sự chăm sóc đặc biệt. Vui lòng xem lại chi tiết cuộc hẹn của bạn dưới đây:</p>
        <h3>Thông tin cuộc hẹn</h3>
        <div><b>Thời gian: </b>${dataSend.time} - ${dataSend.date} </div>
        <div><b>Bác sĩ phụ trách: </b>${dataSend.doctorName}</div>
        <div><b>Địa điểm khám bệnh: </b>${dataSend.clinicName} Tại: ${dataSend.clinicAddress}</div>
        <p>Nếu các thông tin trên là đúng, vui lòng xác nhận và hoàn tất thủ tục tại liên kết bên dưới:</p>
        <div><a href="${dataSend.redirectLink}">Xác nhận</a></div>
        <p>Xin lưu ý các thông tin quan trọng sau:</p>

        <p>Thời gian đến: Vui lòng đến trước giờ hẹn ít nhất 15 phút để hoàn thành mọi thủ tục giấy tờ và đăng ký cần thiết. Điều này sẽ giúp đảm bảo rằng cuộc tư vấn của bạn bắt đầu nhanh chóng.</p>
        <p>      Giấy tờ và Giấy tờ tùy thân: Vui lòng mang theo các giấy tờ sau:</p>
  
        
       <p>Thẻ bảo hiểm y tế của bạn (nếu có)</p>
       
        <p>Bất kỳ hồ sơ y tế, kết quả xét nghiệm hoặc báo cáo hình ảnh có liên quan nào chưa được cung cấp trước đây (nếu có)</p>
       <p><b>Thuốc và Dị ứng: Điều quan trọng là phải thông báo cho nhà cung cấp dịch vụ y tế của bạn về bất kỳ loại thuốc nào bạn đang dùng, bao gồm cả thuốc không kê đơn, thảo dược bổ sung hoặc vitamin. Ngoài ra, vui lòng thông báo cho chúng tôi về bất kỳ dị ứng hoặc phản ứng bất lợi nào đã biết đối với thuốc.</p>
        
       <p><b>Hủy hoặc đổi lịch: </b>Nếu bạn không thể tham dự cuộc hẹn hoặc cần đổi lịch, vui lòng liên hệ với văn phòng của chúng tôi trước ít nhất 24 giờ. Điều này sẽ cho phép chúng tôi cung cấp vị trí cuộc hẹn cho một bệnh nhân khác có nhu cầu.</p>
        
       <p><b>Phạm vi bảo hiểm: </b>Vui lòng đảm bảo rằng bạn đã xem xét phạm vi bảo hiểm sức khỏe của mình và bất kỳ yêu cầu cụ thể nào liên quan đến giới thiệu, ủy quyền trước hoặc đồng thanh toán. Nếu bạn có bất kỳ câu hỏi hoặc thắc mắc nào về bảo hiểm của mình, nhân viên hành chính của chúng tôi sẽ sẵn lòng hỗ trợ bạn.</p>
        
       <p>Nếu bạn có bất kỳ câu hỏi nào hoặc cần thêm thông tin, xin vui lòng liên hệ với chúng tôi tại [Số liên lạc] hoặc qua email tại [Địa chỉ email]. Đội ngũ nhân viên tận tâm của chúng tôi luôn sẵn sàng hỗ trợ bạn và đảm bảo chuyến thăm của bạn diễn ra suôn sẻ.</p>
        
       <p>Chúng tôi chân thành đánh giá cao cơ hội được trở thành nhà cung cấp dịch vụ chăm sóc sức khỏe của bạn và cam kết cung cấp chất lượng chăm sóc y tế cao nhất. Chúng tôi mong được gặp bạn tại cuộc hẹn theo lịch trình của bạn.</p>
        <p>Trân trọng,</p>
        `,
  });
};
let sendCompleteMailToClient = async (dataSend) => {
  let transporter = nodeMailer.createTransport({
    host: "smtp.gmail.com",
    port: 587,
    secure: false,
    auth: {
      user: process.env.EMAIL_APP,
      pass: process.env.EMAIL_APP_PASSWORD,
    },
  });
  let info = await transporter.sendMail({
    from: '"AMIGO HEALTHCARE" <process.env.EMAIL_APP>',
    to: dataSend.receiverEmail,
    subject: "Thông tin đặt lịch khám bệnh",
    html: `
        <h3>Xin chào ${dataSend.clientName} </h3>
        <p> Bạn nhận được email này vì đã hoàn tất lịch hẹn khám bệnh. </p>
        <p> Tóm tắt quá trình khám bệnh: ${dataSend.sum} </p>
        <p> Kế hoạch điều trị: ${dataSend.plan} </p>
        <p> Hướng dẫn về Thuốc: ${dataSend.medical} </p>
        <p> Khuyến nghị về lối sống:: ${dataSend.lifeStyle} </p>
        <p> lịch tái khám:: ${dataSend.appointments} </p>
        `,
    attachments: [
      {
        filename: `${dataSend.clientId}-${new Date().getTime()}.png`,
        content: dataSend.imageBase64.split("base64,")[1],
        encoding: "base64",
      },
    ],
  });

  return true;
};
module.exports = { sendMailToClient, sendCompleteMailToClient };
