import express from "express";
import homeController from "../controllers/homeController";
import userController from "../controllers/userController";
import doctorController from "../controllers/doctorController";
import clientController from "../controllers/clientController";
import clinicController from "../controllers/clinicController";
import specialtyController from "../controllers/specialtyController";
let router = express.Router();

let initWebRoutes = (app) => {
  router.get("/", homeController.getHomePage);
  router.get("/about", homeController.getAboutPage);
  router.get("/crud", homeController.getCRUD);
  router.post("/post-crud", homeController.postCRUD);
  router.get("/get-crud", homeController.displayGetCRUD);
  router.get("/edit-crud", homeController.getEditCRUD);
  router.post("/put-crud", homeController.putCRUD);
  router.get("/delete-crud", homeController.deleteCRUD);

  router.post("/api/login", userController.handleLogin);
  router.get("/api/get-list-user", userController.handleGetUser);
  router.post("/api/create-new-user", userController.handleCreateNewUser);
  router.put("/api/edit-user", userController.handleEditUser);
  router.delete("/api/delete-user", userController.handleDeleteUser);

  router.get("/api/get-all-code", userController.getAllCode);

  //doctor
  router.post(
    "/api/completed-appointment",
    doctorController.completeAppointment
  );
  router.get("/api/get-top-doctor-home", doctorController.getTopDoctorHome);
  router.get("/api/get-all-doctors", doctorController.getAllDoctors);
  router.post("/api/save-info-doctor", doctorController.handleSaveInfoDoctor);
  router.get("/api/get-detail-doctor", doctorController.getDetailDoctorById);
  router.get(
    "/api/get-detail-doctor-by-id",
    doctorController.getDetailDoctorById
  );
  router.post("/api/bulk-create-schedule", doctorController.bulkCreateSchedule);
  router.get(
    "/api/get-schedule-doctor-by-date",
    doctorController.getScheduleByDate
  );
  router.get(
    "/api/get-extra-doctor-info-by-id",
    doctorController.getExtraDoctorInfoById
  );
  router.get(
    "/api/get-profile-doctor-by-id",
    doctorController.getProfileDoctorById
  );
  router.get(
    "/api/get-appointment-by-date",
    doctorController.getAppointmentByDate
  );
  //client
  router.post(
    "/api/patient-book-appointment",
    clientController.bookAppointment
  );
  router.post("/api/verify-booking", clientController.postVerifyBooking);

  //specialty
  router.post("/api/create-specialty", specialtyController.createSpecialty);
  router.get("/api/get-all-specialty", specialtyController.getListSpecialty);
  router.get(
    "/api/get-detail-specialty-by-id",
    specialtyController.getDetailSpecialtyById
  );

  //clinic
  router.post("/api/create-clinic", clinicController.createClinic);
  router.get("/api/get-all-clinic", clinicController.getListClinic);
  router.get(
    "/api/get-detail-clinic-by-id",
    clinicController.getDetailClinicById
  );

  return app.use("/", router);
};

module.exports = initWebRoutes;
