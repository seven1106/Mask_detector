import clientService from "../services/client_service";
let bookAppointment = async (req, res) => {
  try {
    let data = await clientService.postBookAppointment(req.body);
    return res.status(200).json(data);
  } catch (e) {
    console.log(e);
    return res.status(200).json({
      errCode: -1,
      message: "Error from server",
    });
  }
};
let postVerifyBooking = async (req, res) => {
  try {
    let data = await clientService.postVerifyBooking(req.body);
    return res.status(200).json(data);
  } catch (e) {
    console.log(e);
    return res.status(200).json({
      errCode: -1,
      message: "Error from server",
    });
  }
};

module.exports = { bookAppointment, postVerifyBooking };
