import clinicService from "../services/clinic_service";
let createClinic = async (req, res) => {
  try {
    let message = await clinicService.createClinic(req.body);
    res.status(200).json(message);
  } catch (e) {
    console.log(e);
    res.status(200).json({
      errCode: -1,
      errMessage: "Error from server",
    });
  }
};
let getListClinic = async (req, res) => {
  try {
    let response = await clinicService.getAllClinic();
    return res.status(200).json(response);
  } catch (e) {
    console.log(e);
    return res.status(200).json({
      errCode: -1,
      message: "Error from server",
    });
  }
};
let getDetailClinicById = async (req, res) => {
  try {
    let id = req.query.id;
    if (!id) {
      return res.status(200).json({
        errCode: 1,
        errMessage: "Missing required parameter",
      });
    } else {
      let response = await clinicService.getDetailClinicById(id);
      return res.status(200).json(response);
    }
  } catch (e) {
    console.log(e);
    return res.status(200).json({
      errCode: -1,
      message: "Error from server",
    });
  }
};
module.exports = {
  createClinic: createClinic,
  getListClinic: getListClinic,
  getDetailClinicById: getDetailClinicById,
};
