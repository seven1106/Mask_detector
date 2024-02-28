"use strict";
const { Model } = require("sequelize");
module.exports = (sequelize, DataTypes) => {
  class User extends Model {
    /**
     * Helper method for defining associations.
     * This method is not a part of Sequelize lifecycle.
     * The `models/index` file will call this method automatically.
     */
    static associate(models) {
      User.belongsTo(models.allCode, {
        foreignKey: "position",
        targetKey: "key",
        as: "positionData",
      });
      User.belongsTo(models.allCode, {
        foreignKey: "gender",
        targetKey: "key",
        as: "genderData",
      });
      User.hasOne(models.markdown, { foreignKey: "doctorId" });
      User.hasOne(models.detailDoctor, { foreignKey: "doctorId" });
      User.hasMany(models.schedule, {
        foreignKey: "doctorId",
        as: "doctorData",
      });
      User.hasMany(models.booking, {
        foreignKey: "clientId",
        as: "client",
      });
      
    }
  }
  User.init(
    {
      // id: DataTypes.INTEGER,
      email: DataTypes.STRING,
      password: DataTypes.STRING,
      firstName: DataTypes.STRING,
      lastName: DataTypes.STRING,
      address: DataTypes.STRING,
      gender: DataTypes.STRING,
      image: DataTypes.TEXT,
      phoneNumber: DataTypes.STRING,
      roleId: DataTypes.STRING,
      position: DataTypes.STRING,
    },
    {
      sequelize,
      modelName: "User",
    }
  );
  return User;
};
