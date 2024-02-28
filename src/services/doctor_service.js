import e from "express";
import db from "../models";
import _ from "lodash";
import emailService from "./email_service";
require("dotenv").config();

const MAX_NUMBER_SCHEDULE = process.env.MAX_NUMBER_SCHEDULE;
let getTopDoctorHome = (limitInput) => {
  return new Promise(async (resolve, reject) => {
    try {
      let users = await db.User.findAll({
        limit: limitInput,
        where: { roleId: "R2" },
        order: [["createdAt", "DESC"]],
        attributes: {
          exclude: ["password"],
        },
        include: [
          {
            model: db.allCode,
            as: "positionData",
            attributes: ["value_en", "value_vi"],
          },
          {
            model: db.allCode,
            as: "genderData",
            attributes: ["value_en", "value_vi"],
          },
        ],
        raw: true,
        nest: true,
      });
      resolve({
        errCode: 0,
        data: users,
      });
    } catch (e) {
      reject(e);
    }
  });
};
let getAllDoctors = () => {
  return new Promise(async (resolve, reject) => {
    try {
      let doctors = await db.User.findAll({
        where: { roleId: "R2" },
        attributes: {
          exclude: ["password"],
        },
        include: [
          {
            model: db.allCode,
            as: "positionData",
            attributes: ["value_en", "value_vi"],
          },
          {
            model: db.allCode,
            as: "genderData",
            attributes: ["value_en", "value_vi"],
          },
        ],
        raw: true,
        nest: true,
      });
      resolve({
        errCode: 0,
        data: doctors,
      });
    } catch (e) {
      reject(e);
    }
  });
};
let saveDetailInforDoctor = (infor) => {
  return new Promise(async (resolve, reject) => {
    try {
      let detailDoctor = await db.detailDoctor.findOne({
        where: { doctorId: infor.doctorId },
        raw: false,
      });
      if (detailDoctor) {
        (detailDoctor.priceId = infor.selectedPri),
          (detailDoctor.paymentId = infor.selectedPay),
          (detailDoctor.provinceId = infor.selectedPro),
          (detailDoctor.nameClinic = infor.nameClinic),
          (detailDoctor.addressClinic = infor.addressClinic),
          (detailDoctor.note = infor.note),
          (detailDoctor.specialtyId = infor.specialtyId),
          await detailDoctor.save();
      } else {
        await db.detailDoctor.create({
          priceId: infor.selectedPri,
          paymentId: infor.selectedPay,
          provinceId: infor.selectedPro,
          nameClinic: infor.nameClinic,
          addressClinic: infor.addressClinic,
          doctorId: infor.doctorId,
          note: infor.note,
          specialtyId: infor.selectedSpe,
          clinicId: infor.selectedSpe,
        });
      }
      let markdown = await db.markdown.findOne({
        where: { doctorId: infor.doctorId },
        raw: false,
      });

      if (markdown) {
        //update
        (markdown.contentHTML = infor.contentHTML),
          (markdown.contentMarkdown = infor.contentMarkdown),
          (markdown.description = infor.description),
          (markdown.doctorId = infor.doctorId),
          await markdown.save();

        resolve({
          errCode: 0,
          message: "Update the markdown's infor success",
        });
      } else {
        //create
        await db.markdown.create({
          contentHTML: infor.contentHTML,
          contentMarkdown: infor.contentMarkdown,
          description: infor.description,
          doctorId: infor.doctorId,
        });

        resolve({
          errCode: 0,
          message: "Create the doctor's infor success",
        });
      }
    } catch (e) {
      reject(e);
    }
  });
};
let getDetailDoctorById = (id) => {
  return new Promise(async (resolve, reject) => {
    try {
      if (!id) {
        resolve({
          errCode: 1,
          errMessage: "Missing required parameter",
        });
      } else {
        let data = await db.User.findOne({
          where: { id: id },
          attributes: {
            exclude: ["password"],
          },
          include: [
            {
              model: db.allCode,
              as: "positionData",
              attributes: ["value_en", "value_vi"],
            },
            {
              model: db.allCode,
              as: "genderData",
              attributes: ["value_en", "value_vi"],
            },
            {
              model: db.markdown,
              attributes: ["contentHTML", "contentMarkdown", "description"],
            },
            {
              model: db.detailDoctor,

              attributes: {
                exclude: ["id", "doctorId"],
              },
              include: [
                {
                  model: db.allCode,
                  as: "priceData",
                  attributes: ["value_en", "value_vi"],
                },
                {
                  model: db.allCode,
                  as: "paymentData",
                  attributes: ["value_en", "value_vi"],
                },
                {
                  model: db.allCode,
                  as: "provinceData",
                  attributes: ["value_en", "value_vi"],
                },
              ],
            },
          ],
          raw: true,
          nest: true,
        });
        if (data && data.image) {
          data.image = new Buffer(data.image, "base64").toString("binary");
        }
        if (!data) {
          data = {};
        }
        resolve({
          errCode: 0,
          data: data,
        });
      }
    } catch (e) {
      reject(e);
    }
  });
};
let bulkCreateSchedule = (inputData) => {
  return new Promise(async (resolve, reject) => {
    try {
      if (!inputData) {
        resolve({
          errCode: 1,
          errMessage: "Missing required parameter",
        });
      } else {
        let data = inputData;
        if (data && data.length > 0) {
          data = data.map((item) => {
            item.maxNumber = MAX_NUMBER_SCHEDULE;
            return item;
          });
        }
        let exist = await db.schedule.findAll({
          where: {
            doctorId: inputData[0].doctorId,
            date: inputData[0].date,
          },
        });
        if (exist && exist.length > 0) {
          //update
          await db.schedule.destroy({
            where: {
              doctorId: inputData[0].doctorId,
              date: inputData[0].date,
            },
          });
        }
        await db.schedule.bulkCreate(data);
        resolve({
          errCode: 0,
          message: "Create the doctor's schedule success",
        });
      }
    } catch (e) {
      reject(e);
    }
  });
};
let getScheduleByDate = (doctorId, date) => {
  return new Promise(async (resolve, reject) => {
    try {
      if (!doctorId || !date) {
        resolve({
          errCode: 1,
          errMessage: "Missing required parameter",
        });
      } else {
        let data = await db.schedule.findAll({
          where: {
            doctorId: doctorId,
            date: date,
          },
        });
        if (!data) data = [];
        resolve({
          errCode: 0,
          data: data,
        });
      }
    } catch (e) {
      reject(e);
    }
  });
};
let getExtraDoctorInfoById = (id) => {
  return new Promise(async (resolve, reject) => {
    try {
      if (!id) {
        resolve({
          errCode: 1,
          errMessage: "Missing required parameter",
        });
      } else {
        if (id === "ALL") {
          let data = await db.detailDoctor.findAll({});
          if (!data) data = [];
          resolve({
            errCode: 0,
            data: data,
          });
        } else {
          let data = await db.detailDoctor.findOne({
            where: { doctorId: id },
            attributes: {
              exclude: ["password"],
            },
            include: [
              {
                model: db.allCode,
                as: "priceData",
                attributes: ["value_en", "value_vi"],
              },
              {
                model: db.allCode,
                as: "paymentData",
                attributes: ["value_en", "value_vi"],
              },
              {
                model: db.allCode,
                as: "provinceData",
                attributes: ["value_en", "value_vi"],
              },
            ],
            raw: false,
            nest: true,
          });
          if (!data) data = {};
          resolve({
            errCode: 0,
            data: data,
          });
        }
      }
    } catch (e) {
      reject(e);
    }
  });
};
let getProfileDoctorById = (id) => {
  return new Promise(async (resolve, reject) => {
    try {
      if (!id) {
        resolve({
          errCode: 1,
          errMessage: "Missing required parameter",
        });
      } else {
        let data = await db.User.findOne({
          where: { id: id },
          attributes: {
            exclude: ["password"],
          },
          include: [
            {
              model: db.allCode,
              as: "positionData",
              attributes: ["value_en", "value_vi"],
            },

            {
              model: db.markdown,
              attributes: ["contentHTML", "contentMarkdown", "description"],
            },
            {
              model: db.detailDoctor,
              attributes: {
                exclude: ["id", "doctorId"],
              },
              include: [
                {
                  model: db.allCode,
                  as: "priceData",
                  attributes: ["value_en", "value_vi"],
                },
                {
                  model: db.allCode,
                  as: "paymentData",
                  attributes: ["value_en", "value_vi"],
                },
                {
                  model: db.allCode,
                  as: "provinceData",
                  attributes: ["value_en", "value_vi"],
                },
              ],
            },
          ],
          raw: true,
          nest: true,
        });
        if (data && data.image) {
          data.image = new Buffer(data.image, "base64").toString("binary");
        }
        if (!data) {
          data = {};
        }
        resolve({
          errCode: 0,
          data: data,
        });
      }
    } catch (e) {
      reject(e);
    }
  });
};
let getAppointmentByDate = (doctorId, date) => {
  return new Promise(async (resolve, reject) => {
    try {
      if (!doctorId || !date) {
        resolve({
          errCode: 1,
          errMessage: "Missing required parameter",
        });
      } else {
        if (doctorId === "ALL") {
          let data = await db.booking.findAll({
            where: {
              date: date,
            },
            include: [
              {
                model: db.User,
                as: "client",
                attributes: ["email", "firstName", "gender", "address"],
                include: [
                  {
                    model: db.allCode,
                    as: "genderData",
                    attributes: ["value_en", "value_vi"],
                  },
                ],
              },
            ],
            raw: false,
            nest: true,
          });
          if (!data) data = [];
          resolve({
            errCode: 0,
            data: data,
          });
        } else {
          let data = await db.booking.findAll({
            where: {
              doctorId: doctorId,
              date: date,
              statusId: "S2",
            },
            include: [
              {
                model: db.User,
                as: "client",
                attributes: ["email", "firstName", "address"],
                include: [
                  {
                    model: db.allCode,
                    as: "genderData",
                    attributes: ["value_en"],
                  },
                ],
              },
            ],
            raw: false,
            nest: true,
          });
          if (!data) data = [];
          resolve({
            errCode: 0,
            data: data,
          });
        }
      }
    } catch (e) {
      reject(e);
    }
  });
};
let completeAppointment = (data, type) => {
  return new Promise(async (resolve, reject) => {
    try {
      if (!data.doctorId || !data.sum) {
        resolve({
          errCode: 1,
          errMessage: "Missing required parameter",
        });
      } else {
        let appointment = await db.booking.findOne({
          where: {
            clientId: data.clientId,
            statusId: "S2",
            doctorId: data.doctorId,
          },
        });
        if (appointment) {
          if (type === "SEND") {
            let resMail = await emailService.sendCompleteMailToClient({
              receiverEmail: data.email,
              clientName: data.clientName,
              sum: data.sum,
              plan: data.plan,
              medical: data.medical,
              lifeStyle: data.lifeStyle,
              appointments: data.appointments,
              imageBase64: data.imageBase64,
            });
            if (resMail === true) {
              appointment.statusId = "S3";
              await appointment.save();
              resolve({
                errCode: 0,
                message: "Compelete appointment success",
              });
            } else {
              resolve({
                errCode: 0,
                message: "Send mail fail",
              });
            }
          } else {
            appointment.statusId = "S3";
            await appointment.save();
          }

          resolve({
            errCode: 0,
            message: "Compelete appointment success",
          });
        } else {
          resolve({
            errCode: 2,
            errMessage: "Appointment not found",
          });
        }
      }
    } catch (e) {
      reject(e);
    }
  });
};

module.exports = {
  getTopDoctorHome: getTopDoctorHome,
  getAllDoctors: getAllDoctors,
  saveDetailInforDoctor: saveDetailInforDoctor,
  getDetailDoctorById: getDetailDoctorById,
  bulkCreateSchedule: bulkCreateSchedule,
  getScheduleByDate: getScheduleByDate,
  getExtraDoctorInfoById: getExtraDoctorInfoById,
  getProfileDoctorById: getProfileDoctorById,
  getAppointmentByDate: getAppointmentByDate,
  completeAppointment: completeAppointment,
};
