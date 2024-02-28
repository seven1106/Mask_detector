import CRUD_service from "../services/CRUD_service";
import db from "../models";

let getHomePage = async (req, res) => {
  try {
    let data = await db.User.findAll();
    return res.render("homepage.ejs", {
      data: JSON.stringify(data),
    });
  } catch (err) {
    console.log(err);
  }
};

let getAboutPage = (req, res) => {
  return res.render("test/about.ejs");
};
let getCRUD = (req, res) => {
  return res.render("test/CRUD.ejs");
};
let postCRUD = async (req, res) => {
  let createUser = await CRUD_service.createNewUser(req.body);
  console.log(createUser);
  console.log(req.body);
  return res.send("Post CRUD from server");
};
let displayGetCRUD = async (req, res) => {
  let data = await CRUD_service.getAllUser();
  console.log(data);
  return res.render("test/displayCRUD.ejs", {
    dataTable: data,
  });
};
let getEditCRUD = async (req, res) => {
  let userId = req.query.id;
  if (userId) {
    let userData = await CRUD_service.getUserInfoById(userId);
    return res.render("test/editCRUD.ejs", {
      user: userData,
    });
  } else {
    return res.send("User not found");
  }
};
let putCRUD = async (req, res) => {
  let data = req.body;
  if (data) {
    let allUsers = await CRUD_service.updateUserData(data);
    return res.render("test/displayCRUD.ejs", {
      dataTable: allUsers,
    });
  } else {
    return res.send("User not found");
  }
};
let deleteCRUD = async (req, res) => {
  let id = req.query.id;
  if (id) {
    let deleteUser = await CRUD_service.deleteUserData(id);
    return res.render("test/displayCRUD.ejs", {
      dataTable: deleteUser,
    });
  } else {
    return res.send("User not found");
  }
};

module.exports = {
  getHomePage: getHomePage,
  getAboutPage: getAboutPage,
  getCRUD: getCRUD,
  postCRUD: postCRUD,
  displayGetCRUD: displayGetCRUD,
  getEditCRUD: getEditCRUD,
  putCRUD: putCRUD,
  deleteCRUD: deleteCRUD,
};
