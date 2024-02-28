"use strict";
const { Model } = require("sequelize");
module.exports = (sequelize, DataTypes) => {
  class booking extends Model {
    /**
     * Helper method for defining associations.
     * This method is not a part of Sequelize lifecycle.
     * The `models/index` file will call this method automatically.
     */
    static associate(models) {
      booking.belongsTo(models.User, {
        foreignKey: "clientId",
        targetKey: "id",
        as: "client",
      });
    }
  }
  booking.init(
    {
      statusId: DataTypes.INTEGER,
      doctorId: DataTypes.INTEGER,
      clientId: DataTypes.INTEGER,
      date: DataTypes.STRING,
      timeType: DataTypes.STRING,
      token: DataTypes.STRING,
    },
    {
      sequelize,
      modelName: "booking",
    }
  );
  return booking;
};
