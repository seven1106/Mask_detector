import bcrypt from "bcryptjs";
import db from "../models";
var salt = bcrypt.genSaltSync(10);
// Store hash in your password DB.
let createNewUser = async (user) => {
  return new Promise(async (resolve, reject) => {
    try {
      let hashPasswordFromBcrypt = await hashUserPassword(user.password);
      await db.User.create({
        email: user.email,
        password: hashPasswordFromBcrypt,
        firstName: user.firstName,
        lastName: user.lastName,
        address: user.address,
        gender: user.gender === "1" ? true : false,
        phoneNumber: user.phoneNumber,
        roleId: user.roleId,
      });
      resolve("ok! create new user success.");
    } catch (error) {
      reject(error);
    }
  });
};
let hashUserPassword = (password) => {
  return new Promise(async (resolve, reject) => {
    try {
      let hashPassword = await bcrypt.hashSync(password, salt);
      resolve(hashPassword);
    } catch (error) {
      reject(error);
    }
  });
};
let getAllUser = () => {
  return new Promise(async (resolve, reject) => {
    try {
      let users = await db.User.findAll();
      resolve(users);
    } catch (error) {
      reject(error);
    }
  });
};
let getUserInfoById = (userId) => {
  return new Promise(async (resolve, reject) => {
    try {
      let user = await db.User.findOne({
        where: { id: userId },
        raw: true,
      });
      if (user) {
        resolve(user);
      }
    } catch (error) {
      reject(error);
    }
  });
};
let updateUserData = async (data) => {
  return new Promise(async (resolve, reject) => {
    try {
      let user = await db.User.findOne({
        where: { id: data.id },
      });
      if (user) {
        user.firstName = data.firstName;
        user.lastName = data.lastName;
        user.address = data.address;
        await user.save();
        let allUsers = await db.User.findAll();
        resolve(allUsers);
      } else {
        return "User not found";
      }
    } catch (error) {
      console.log(error);
    }
  });
};
let deleteUserData = (userId) => {
  return new Promise(async (resolve, reject) => {
    let user = await db.User.findOne({
      where: { id: userId },
    });
    if (user) {
      await user.destroy();
      let deleteUser = await db.User.findAll();
      resolve(deleteUser);
    } else {
      reject("User not found");
    }
  });
};

module.exports = {
  createNewUser: createNewUser,
  getAllUser: getAllUser,
  getUserInfoById: getUserInfoById,
  updateUserData: updateUserData,
  deleteUserData: deleteUserData,
};
