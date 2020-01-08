function generate_name() {
  var style = ["Nike Air", "Air Jordan", "Nike", "Nike React"];
  var type = ["Force", "Retro", "Zoom", "Pegasus", "ZoomX", "Vaporfly", "Infinity", "VaporMax", "Max"];

  style_id = Math.floor(Math.random() * style.length);
  type_id = Math.floor(Math.random() * type.length);
  version_no = Math.floor(Math.random() * 14);

  return style[style_id] + " " + type[type_id] + (version_no !== 0 ? " " + version_no : "");
}

function generate_price() {
  var price = Math.floor(Math.random() * 350) + 49.99;
  return "$" + price.toFixed(2);
}

function load_card(image) {
  var popover_msg = "This shoe card is just for show (the link doesn't go anywhere)";
  var html = '' +
    '<div class="col-lg-3 col-md-4 col-sm-6 mb-4">' +
      '<div class="card h-100">' +
        '<a href="#" data-toggle="popover" data-trigger="focus" data-content="' + popover_msg + '">' +
          '<img class="card-img-top" src="./images/' + image + '" alt="">' +
        '</a>' +
        '<div class="card-body">' +
          '<h4 class="card-title">' +
            '<a href="#" data-toggle="popover" data-trigger="focus" data-content="' + popover_msg + '">' + generate_name() + '</a>' +
          '</h4>' +
          '<h5>' + generate_price() + '</h5>' +
        '</div>' +
      '</div>' +
    '</div>';

  $(html).insertBefore("#generate");
  $('#shoe-cards > div:nth-last-child(2) [data-toggle="popover"]').popover();
}

function display_warning() {
  var warning_msg = "A new server is starting, this may take slightly longer"
  $("#generate button").removeClass("btn-success").removeClass("btn-danger").addClass("btn-warning");
  $("#generate button").html(
    '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> ' + warning_msg
  );
}

function display_error() {
  $("#generate button").prop("disabled", false);
  $("#generate button").html("An error occurred. Click to try again.");
  $("#generate button").removeClass("btn-success").removeClass("btn-warning").addClass("btn-danger");
}

function send_image_request(errback) {
  $.ajax({
    timeout: 30000,
    type: "GET",
    async: false,
    url: "https://jpotdr65xi.execute-api.us-west-2.amazonaws.com/default/shoeGanGenerate",
    success: function (result) {
      load_card(result["images"][0]["name"]);
      $("#generate button").prop("disabled", false);
      $("#generate button").html("Generate!");
      $("#generate button").removeClass("btn-danger").removeClass("btn-warning").addClass("btn-success");
    },
    error: function (result) {
      errback();
    }
  });
}

function generate_image(num_tries) {
  if (num_tries > 1) {
    send_image_request(function () {
      display_warning();
      setTimeout(function () {
        generate_image(num_tries - 1);
      }, 30000)
    });
  }
  else {
    send_image_request(function () {
      display_error();
    });
  }
}

$(document).ready(function () {
  $("#generate button").click(function () {
    $(this).prop("disabled", true);
    $(this).html(
      '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...'
    );

    // Retry image generation if lambda cold start causes a timeout
    generate_image(2);
  });
});
