"use strict";
/** @type {import('sequelize-cli').Migration} */
module.exports = {
  async up(queryInterface, Sequelize) {
    await queryInterface.createTable("markdowns", {
      id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER,
      },
      contentHTML: {
        type: Sequelize.TEXT({ length: "long" }),
      },
      contentMarkdown: {
        type: Sequelize.TEXT({ length: "long" }),
      },
      description: {
        type: Sequelize.TEXT,
      },
      doctorId: {
        type: Sequelize.INTEGER,
      },
     
      specialtyId: {
        type: Sequelize.INTEGER,
      },
     
      clinicId: {
        type: Sequelize.INTEGER,
      },
     
      createdAt: {
        allowNull: false,
        type: Sequelize.DATE,
      },
      updatedAt: {
        allowNull: false,
        type: Sequelize.DATE,
      },
    });
  },
  async down(queryInterface, Sequelize) {
    await queryInterface.dropTable("markdowns");
  },
};
