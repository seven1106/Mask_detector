"use strict";
const { Model } = require("sequelize");
module.exports = (sequelize, DataTypes) => {
  class allCode extends Model {
    /**
     * Helper method for defining associations.
     * This method is not a part of Sequelize lifecycle.
     * The `models/index` file will call this method automatically.
     */
    static associate(models) {
      allCode.hasMany(models.User, {
        foreignKey: "position",
        as: "positionData",
      });
      allCode.hasMany(models.User, {
        foreignKey: "gender",
        as: "genderData",
      });
      allCode.hasMany(models.detailDoctor, {
        foreignKey: "priceId",
        as: "priceData",
      });
      allCode.hasMany(models.detailDoctor, {
        foreignKey: "provinceId",
        as: "provinceData",
      });
      allCode.hasMany(models.detailDoctor, {
        foreignKey: "paymentId",
        as: "paymentData",
      });
    }
  }
  allCode.init(
    {
      key: DataTypes.STRING,
      type: DataTypes.STRING,
      value_en: DataTypes.STRING,
      value_vi: DataTypes.STRING,
    },
    {
      sequelize,
      modelName: "allCode",
    }
  );
  return allCode;
};
