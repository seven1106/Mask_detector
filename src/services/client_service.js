import db from "../models";
import emailService from "./email_service";
import _ from "lodash";
import { v4 as uuidv4 } from "uuid";
require("dotenv").config();
let postBookAppointment = (data) => {
  return new Promise(async (resolve, reject) => {
    try {
      if (!data.email || !data.doctorId) {
        console.log("Missing required parameter!", data);
        resolve({
          errCode: 1,
          errMessage: "Missing required parameter!",
        });
      } else {
        let token = uuidv4();
        await emailService.sendMailToClient({
          receiverEmail: data.email,
          clientName: data.fullName,
          date: data.date,
          time: data.timeType,
          doctorName: data.doctorName,
          clinicName: data.clinicName,
          clinicAddress: data.clinicAddress,
          redirectLink: buildUrlEmail(data.doctorId, token),
        });
        let user = await db.User.findOrCreate({
          where: { email: data.email },
          defaults: {
            email: data.email,
            roleId: "R3",
            firstName: data.fullName,
            address: data.address,
            gender:data.selectedGender,
          },
        });
        console.log("check use", user);
        if (user && user[0]) {
          await db.booking.create({
           
              statusId: "S1",
              doctorId: data.doctorId,
              date: data.date,
              timeType: data.timeType,
              clientId: user[0].id,
              token: token,
          });
          resolve({
            errCode: 0,
            errMessage: "Book appointment success!",
          });
        } else {
          resolve({
            errCode: 2,
            errMessage: "Cannot find user!",
          });
        }
      }
    } catch (e) {
      reject(e);
    }
  });
};
let buildUrlEmail = (doctorId, token) => {
  let result = `${process.env.URL_REACT}/verify-booking?token=${token}&doctorId=${doctorId}`;
  return result;
};
let postVerifyBooking = (data) => {
  return new Promise(async (resolve, reject) => {
    try {
      if (!data.token || !data.doctorId) {
        console.log("Missing required parameter!", data);
        resolve({
          errCode: 1,
          errMessage: "Missing required parameter!",
        });
      } else {
        let booking = await db.booking.findOne({
          where: {
            token: data.token,
            doctorId: data.doctorId,
            statusId: "S1",
          },
        });
        if (booking) {
          await db.booking.update(
            {
              statusId: "S2",
            },
            {
              where: {
                token: data.token,
                doctorId: data.doctorId,
              },
            }
          );
          resolve({
            errCode: 0,
            errMessage: "Verify booking success!",
          });
        } else {
          resolve({
            errCode: 2,
            errMessage: "Cannot find booking!",
          });
        }
      }
    } catch (e) {
      reject(e);
    }
  });
};

module.exports = {
  postBookAppointment: postBookAppointment,
  buildUrlEmail: buildUrlEmail,
  postVerifyBooking: postVerifyBooking,
};
