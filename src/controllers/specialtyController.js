import specialtyService from "../services/specialty_service";
let createSpecialty = async (req, res) => {
  try {
    let message = await specialtyService.createSpecialty(req.body);
    res.status(200).json(message);
  } catch (e) {
    console.log(e);
    res.status(200).json({
      errCode: -1,
      errMessage: "Error from server",
    });
  }
};
let getListSpecialty = async (req, res) => {
  try {
    let response = await specialtyService.getAllSpecialty();
    return res.status(200).json(response);
  } catch (e) {
    console.log(e);
    return res.status(200).json({
      errCode: -1,
      message: "Error from server",
    });
  }
};
let getDetailSpecialtyById = async (req, res) => {
  try {
    let id = req.query.id;
    if (!id) {
      return res.status(200).json({
        errCode: 1,
        errMessage: "Missing required parameter",
      });
    } else {
      let response = await specialtyService.getDetailSpecialtyById(id);
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
  createSpecialty: createSpecialty,
  getListSpecialty: getListSpecialty,
  getDetailSpecialtyById: getDetailSpecialtyById,
};
